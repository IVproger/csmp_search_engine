from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st
from matchms.importing import load_from_json, load_from_mgf, load_from_msp
from pymzml.run import Reader
from rdkit import Chem
from rdkit.Chem import Draw


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".mzml": "mzML",
    ".mgf": "MGF",
    ".json": "JSON",
    ".msp": "MSP",
}

API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://fastapi_service:8000").rstrip("/")


class SpectrumParserError(ValueError):
    pass


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1400px;
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }

            .small-note {
                color: rgba(140, 140, 140, 1);
                font-size: 0.95rem;
            }

            .molecule-meta {
                font-size: 1rem;
                line-height: 1.5;
            }

            .spectrum-summary {
                border-left: 4px solid #1f77b4;
                padding: 0.45rem 0.8rem;
                margin: 0.35rem 0 0.75rem 0;
                background: rgba(31, 119, 180, 0.08);
            }

            .candidate-head {
                border-left: 3px solid #2ca02c;
                padding-left: 0.6rem;
                margin: 0.65rem 0 0.45rem 0;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value in (None, ""):
            continue
        return value
    return None


def _extract_matchms_precursor_mz(metadata: dict[str, Any]) -> float | None:
    for key in ("precursor_mz", "precursor mz", "pepmass"):
        value = metadata.get(key)
        if value is None:
            continue

        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def _extract_mzml_precursor_mz(spec: Any) -> float | None:
    selected_precursors = getattr(spec, "selected_precursors", None) or []
    if not selected_precursors:
        return None

    precursor_mz = selected_precursors[0].get("mz")
    if precursor_mz is None:
        return None

    try:
        return float(precursor_mz)
    except (TypeError, ValueError):
        return None


def _parse_mzml(path: Path) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    run = Reader(str(path))

    for index, spec in enumerate(run):
        peaks = [
            (float(mz), float(intensity))
            for mz, intensity in spec.peaks(peak_type="centroided")
        ]
        parsed.append(
            {
                "spectrum_id": str(spec.ID or index),
                "precursor_mz": _extract_mzml_precursor_mz(spec),
                "peaks": peaks,
            }
        )

    return parsed


def _parse_matchms(path: Path, extension: str) -> list[dict[str, Any]]:
    loader_map = {
        ".mgf": load_from_mgf,
        ".msp": load_from_msp,
        ".json": load_from_json,
    }
    spectra = loader_map[extension](str(path))

    parsed: list[dict[str, Any]] = []
    for index, spectrum in enumerate(spectra):
        metadata = spectrum.metadata
        peaks = [
            (float(mz), float(intensity))
            for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities)
        ]
        parsed.append(
            {
                "spectrum_id": str(
                    _first_non_empty(
                        metadata.get("spectrum_id"),
                        metadata.get("id"),
                        metadata.get("scans"),
                        metadata.get("title"),
                        index,
                    )
                ),
                "precursor_mz": _extract_matchms_precursor_mz(metadata),
                "peaks": peaks,
            }
        )

    return parsed


def parse_spectra_for_preview(file_name: str, file_bytes: bytes) -> list[dict[str, Any]]:
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise SpectrumParserError("Unsupported file extension.")

    if not file_bytes:
        raise SpectrumParserError("Uploaded file is empty.")

    with NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name)

    try:
        if extension == ".mzml":
            parsed = _parse_mzml(temp_path)
        else:
            parsed = _parse_matchms(temp_path, extension)
    except Exception as error:
        raise SpectrumParserError(f"Failed to parse file: {error}") from error
    finally:
        temp_path.unlink(missing_ok=True)

    if not parsed:
        raise SpectrumParserError("No spectra were parsed from file.")

    return parsed


def build_spectrum_figure(spectrum: dict[str, Any]) -> go.Figure:
    peaks: list[tuple[float, float]] = spectrum.get("peaks", [])
    if not peaks:
        fig = go.Figure()
        fig.update_layout(
            height=430,
            margin=dict(l=40, r=20, t=40, b=40),
            title="No peaks found for this spectrum",
        )
        return fig

    x_points: list[float | None] = []
    y_points: list[float | None] = []
    for mz, intensity in peaks:
        x_points.extend([mz, mz, None])
        y_points.extend([0.0, intensity, None])

    fig = go.Figure(
        go.Scatter(
            x=x_points,
            y=y_points,
            mode="lines",
            line=dict(color="#1f77b4", width=1.6),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(l=44, r=24, t=50, b=44),
        xaxis_title="m/z",
        yaxis_title="Intensity",
        template="plotly_white",
        title=f"Spectrum {spectrum.get('spectrum_id', '')}",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.35)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.35)")
    return fig


def smiles_to_image(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Draw.MolToImage(molecule, size=(520, 320))


def call_annotation_api(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/annotate-spectrum",
        files={"file": (file_name, file_bytes, "application/octet-stream")},
        timeout=180,
    )

    if response.status_code >= 400:
        detail = response.text
        try:
            payload = response.json()
            detail = payload.get("detail") or payload
        except ValueError:
            pass
        raise RuntimeError(f"API request failed ({response.status_code}): {detail}")

    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError("API returned invalid JSON response.") from error


def render_spectrum_results(response_json: dict[str, Any]) -> None:
    results = response_json.get("results", [])
    if not results:
        st.info("No spectrum results returned by API.")
        return

    st.subheader("Search results")
    st.caption(response_json.get("message", ""))

    response_payload = json.dumps(response_json, indent=2, ensure_ascii=False).encode("utf-8")
    download_name = f"annotation_{response_json.get('file_name', 'results')}.json"
    st.download_button(
        label="Download full annotation JSON",
        data=response_payload,
        file_name=download_name,
        mime="application/json",
        use_container_width=True,
    )
    st.caption("Candidate display settings")
    show_candidate_details = st.toggle("Show candidate details", value=True)

    for spectrum_index, spectrum in enumerate(results):
        spectrum_id = spectrum.get("spectrum_id", "unknown")
        precursor_mz = spectrum.get("precursor_mz")
        spectrum_message = spectrum.get("message")
        candidates = spectrum.get("candidates") or []

        st.markdown(f"### Spectrum {spectrum_id}")
        st.markdown(
            f"""
            <div class="spectrum-summary">
                <b>Precursor m/z:</b> {precursor_mz if precursor_mz is not None else 'N/A'} &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>Candidates:</b> {len(candidates)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if spectrum_message:
            st.markdown(f"<span class='small-note'>{spectrum_message}</span>", unsafe_allow_html=True)

        if not candidates:
            st.info("No candidates available for this spectrum.")
        else:
            summary_rows = [
                {
                    "#": index + 1,
                    "Mass": candidate.get("mass", "N/A"),
                    "Score": candidate.get("similarity_score", "N/A"),
                    "SMILES": str(candidate.get("smiles", "")),
                }
                for index, candidate in enumerate(candidates)
            ]
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)

            if show_candidate_details:
                for index, candidate in enumerate(candidates):
                    st.markdown(
                        f"<div class='candidate-head'>Candidate {index + 1}</div>",
                        unsafe_allow_html=True,
                    )
                    image_col, info_col = st.columns([1.6, 1.4])

                    candidate_smiles = str(candidate.get("smiles", ""))
                    with image_col:
                        structure_image = smiles_to_image(candidate_smiles)
                        if structure_image is not None:
                            st.image(structure_image, use_container_width=True)
                        else:
                            st.warning("Failed to render structure from SMILES.")

                    with info_col:
                        st.markdown(
                            f"""
                            <div class="molecule-meta">
                                <b>Monoisotopic mass</b><br>{candidate.get('mass', 'N/A')}<br><br>
                                <b>Similarity score</b><br>{candidate.get('similarity_score', 'N/A')}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)
                        st.markdown("**SMILES:**")
                        st.code(candidate_smiles)

                    if index < len(candidates) - 1:
                        st.divider()

        spectrum_payload = {
            "status": response_json.get("status"),
            "file_name": response_json.get("file_name"),
            "file_type": response_json.get("file_type"),
            "spectrum": spectrum,
        }
        st.download_button(
            label=f"Download JSON for spectrum {spectrum_id}",
            data=json.dumps(spectrum_payload, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"spectrum_{spectrum_id}.json",
            mime="application/json",
            key=f"download_spectrum_{spectrum_index}_{spectrum_id}",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="CSMP Spectrum Search",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_styles()

    st.title("CSMP Spectrum → Molecule Search")
    st.caption("Upload a spectrum file, preview spectra, and search for candidate molecules.")

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["mzML", "mgf", "json", "msp"],
        help="Supported formats: mzML, MGF, JSON, MSP",
        label_visibility="visible",
    )
    st.markdown(
        "<span class='small-note'>Supported formats: .mzML, .MGF, .JSON, .MSP</span>",
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        return

    file_bytes = uploaded_file.getvalue()
    upload_signature = f"{uploaded_file.name}:{len(file_bytes)}"
    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state["upload_signature"] = upload_signature
        st.session_state.pop("annotation_response", None)

    try:
        parsed_spectra = parse_spectra_for_preview(uploaded_file.name, file_bytes)
    except SpectrumParserError as error:
        st.error(str(error))
        return

    st.subheader("Spectrum preview")
    show_preview = st.toggle("Show spectra preview", value=True)
    st.caption(f"Parsed spectra: {len(parsed_spectra)}")

    if show_preview:
        for index in range(0, len(parsed_spectra), 2):
            row_cols = st.columns(2)
            left_spectrum = parsed_spectra[index]
            right_spectrum = parsed_spectra[index + 1] if index + 1 < len(parsed_spectra) else None

            with row_cols[0]:
                st.plotly_chart(
                    build_spectrum_figure(left_spectrum),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.caption(
                    f"Spectrum ID: {left_spectrum.get('spectrum_id', 'N/A')} | "
                    f"Precursor m/z: {left_spectrum.get('precursor_mz', 'N/A')}"
                )

            with row_cols[1]:
                if right_spectrum is not None:
                    st.plotly_chart(
                        build_spectrum_figure(right_spectrum),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.caption(
                        f"Spectrum ID: {right_spectrum.get('spectrum_id', 'N/A')} | "
                        f"Precursor m/z: {right_spectrum.get('precursor_mz', 'N/A')}"
                    )

    search_col_left, search_col_mid, search_col_right = st.columns([1, 2, 1])
    with search_col_mid:
        run_search = st.button("Search for candidate molecules", use_container_width=True)

    if run_search:
        with st.spinner("Running inference and searching molecular database..."):
            try:
                response_json = call_annotation_api(uploaded_file.name, file_bytes)
            except Exception as error:
                st.error(str(error))
                return

            st.session_state["annotation_response"] = response_json

    if "annotation_response" in st.session_state:
        render_spectrum_results(st.session_state["annotation_response"])


if __name__ == "__main__":
    main()