from pathlib import Path
import pandas as pd
import re
import math

PROTON_MASS = 1.007276466812
_ADDUCT_LOOKUP_PATH = Path(__file__).with_name("adducts_lookup_table.csv")
_adducts_lookup_df = pd.read_csv(_ADDUCT_LOOKUP_PATH, usecols=["adduct", "n_mer", "charge", "mass_shift"])
_adducts_lookup_df = _adducts_lookup_df.dropna(subset=["adduct", "n_mer", "charge", "mass_shift"]).copy()
_adducts_lookup_df["n_mer"] = pd.to_numeric(_adducts_lookup_df["n_mer"], errors="coerce")
_adducts_lookup_df["charge"] = pd.to_numeric(_adducts_lookup_df["charge"], errors="coerce")
_adducts_lookup_df["mass_shift"] = pd.to_numeric(_adducts_lookup_df["mass_shift"], errors="coerce")
_adducts_lookup_df = _adducts_lookup_df.dropna(subset=["n_mer", "charge", "mass_shift"])

adducts_look_up_dict = {
    str(row["adduct"]): {
        "n_mer": float(row["n_mer"]),
        "charge": int(row["charge"]),
        "mass_shift": float(row["mass_shift"]),
    }
    for _, row in _adducts_lookup_df.iterrows()
}


def normalize_adduct(adduct_str):
    if not isinstance(adduct_str, str):
        return '[M]+'
    
    original = adduct_str.strip()
    if original in ['Unknown', '?', '', 'N/A', 'null', 'None']:
        return '[M]+'
    
    result = original
    
    special_cases_before = {
        'M-e': '[M]+',
        '[M+15]+': '[M]+15',
        'M++': '[M]2+',
        'M--': '[M]2-',
        'M-H1': '[M-H]-',
        'M+CH3COOH-H': '[M+CH3COO]-',
        'M+TFA-H': '[M+CF3COO]-',
        'M+FA-H': '[M+HCOO]-',
        'M+H3N+H': '[M+NH4]+',
        '[M+H3N+H]+': '[M+NH4]+',
        'M-H4O2+H': '[M+H-2H2O]+',
        '[M-H4O2+H]+': '[M+H-2H2O]+',
        'M-Ac-H-': '[M+CH3CO-H]-',
        'M-HAc': '[M+CH3CO-H]',
        'M+': '[M]+',
        'M-': '[M]-',
        'M+H+Na': '[M+Na+H]+',
        'M-H+Na': '[M+Na-H]+',
        'M-H+2Na': '[M+2Na-H]+',
        'M-H+Cl': '[M+Cl-H]-',
        'M+NH5': '[M+NH5]+',
        '2M-2H+Na': '[2M+Na-2H]+',
        '[2M-2H+Na]': '[2M+Na-2H]+',
    }
    
    if result in special_cases_before:
        return special_cases_before[result]
    
    synonym_map = {
        'FA': 'HCOO',
        'formate': 'HCOO',
        'acetate': 'CH3COO',
        'TFA': 'CF3COO',
        'ACN': 'CH3CN',
        'Ac': 'CH3CO',
        'HAc': 'CH3COOH',
        'H3N': 'NH3',
        'H4O2': '2H2O',
    }
    
    for old, new in synonym_map.items():
        result = re.sub(r'\b' + re.escape(old) + r'\b', new, result)
    
    def flatten_brackets(text):
        while '(' in text:
            match = re.search(r'(\d*)\(([^)]+)\)', text)
            if not match:
                break
            count = match.group(1) or ''
            formula = match.group(2)
            replacement = formula if not count else f'{count}{formula}'
            text = text.replace(match.group(0), replacement, 1)
        return text
    
    result = flatten_brackets(result)
    
    has_brackets = '[' in result and ']' in result
    
    if has_brackets:
        start = result.find('[')
        end = result.find(']')
        if start < end:
            core = result[start+1:end]
            rest = result[end+1:]
        else:
            core = result.replace('[', '').replace(']', '')
            rest = ''
    else:
        core = result
        rest = ''
    
    def clean_and_reorder(text):
        if not text:
            return text
        
        text = text.replace('*', '')
        
        patterns = [
            (r'\+Cl-$', '+Cl'),
            (r'\+FA-$', '+HCOO'),
            (r'\+Br-$', '+Br'),
            (r'H3N\+H', 'NH4'),
            (r'NH3\+H', 'NH4'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        if text.startswith('M'):
            parts = re.findall(r'([\+\-]?[^\+-]+)', text)
            
            m_part = parts[0] if parts else ''
            other_parts = parts[1:] if len(parts) > 1 else []
            
            if m_part and other_parts:
                plus_parts = [p for p in other_parts if p.startswith('+')]
                minus_parts = [p for p in other_parts if p.startswith('-')]
                neutral_parts = [p for p in other_parts if not p.startswith('+') and not p.startswith('-')]
                
                sorted_parts = plus_parts + neutral_parts + minus_parts
                return m_part + ''.join(sorted_parts)
        
        return text
    
    core = clean_and_reorder(core)
    
    def get_charge_from_rest(rest_text):
        if not rest_text:
            return ''
        
        charge = rest_text.replace('*', '')
        
        if charge in ['+', '-', '2+', '2-', '3+', '3-']:
            return charge
        
        if charge == '++':
            return '2+'
        if charge == '--':
            return '2-'
        
        if re.match(r'^\d+[\+\-]$', charge):
            return charge
        
        return ''
    
    def get_charge_from_core(core_text):
        if not core_text:
            return ''
        
        text = core_text.lower()
        
        charge_map = {
            'm-2h': '2-',
            'm+2h': '2+',
            'm+3h': '3+',
            'm+2na': '2+',
            'm+ca': '2+',
            'm+mg': '2+',
            'm+fe': '2+',
        }
        
        for pattern, charge in charge_map.items():
            if pattern in text:
                return charge
        
        if text.endswith(('+h', '+na', '+k', '+li', '+nh4', '+nh3', '+ch3oh', '+ch3cn')):
            return '+'
        
        if text.endswith(('-h', '+cl', '+br', '+hcoo', '+ch3coo', '+cf3coo', '+cooh')):
            return '-'
        
        if '+h' in text and any(x in text for x in ['-h2o', '-co2', '-co']):
            return '+'
        
        if any(pattern in text for pattern in ['+na-h', '+2na-h', '+k-h']):
            return '+'
        
        if '+cl-h' in text or '+br-h' in text:
            return '-'
        
        if '2m+na-2h' in text:
            return '+'
        
        if re.match(r'^\d+m-', text):
            return '-'
        
        if re.match(r'^\d+m\+', text):
            return '+'
        
        if text.endswith('-h2o') or text.endswith('-co2') or text.endswith('-co'):
            return ''
        
        return ''
    
    charge_from_rest = get_charge_from_rest(rest)
    charge_from_core = get_charge_from_core(core)
    
    charge = charge_from_rest or charge_from_core
    
    def final_processing(core_text, chrg):
        text = core_text
        
        replacements = [
            ('2M-2H+Na', '2M+Na-2H'),
            ('M-H+2Na', 'M+2Na-H'),
            ('M-H+Na', 'M+Na-H'),
            ('M-H+Cl', 'M+Cl-H'),
            ('M-H2+H', 'M+H-H2'),
            ('M+H+Na', 'M+Na+H'),
            ('M+H+CH3OH', 'M+CH3OH+H'),
            ('M+H+CH3CN', 'M+CH3CN+H'),
            ('M+H+HCOOH', 'M+HCOOH+H'),
            ('M+H+C2H6OS', 'M+C2H6OS+H'),
            ('2M-H2O+H', '2M+H-H2O'),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        if 'H2O-H' in text and chrg == '-':
            text = text.replace('H2O-H', '-H-H2O')
        
        if 'H2O+H' in text and chrg == '+':
            text = text.replace('H2O+H', '+H-H2O')
        
        if text.endswith(']'):
            text = text[:-1]
        
        if text == 'M-H2O-H':
            text = 'M-H-H2O'
        
        if text == 'M--H-H2O':
            text = 'M-H-H2O'
        
        if text == 'M+Cl-H':
            text = 'M+Cl-H'
        
        return text
    
    core = final_processing(core, charge)
    
    if core.startswith('['):
        normalized = core
    else:
        normalized = f'[{core}]'
    
    if charge and not normalized.endswith(charge):
        if normalized.endswith(']'):
            normalized += charge
        else:
            normalized = f'[{core}]{charge}'
    
    normalized = re.sub(r'\[\[(.*?)\]\]', r'[\1]', normalized)
    
    fix_map = {
        '[M+H]': '[M+H]+',
        '[M-H]': '[M-H]-',
        '[M+Na]': '[M+Na]+',
        '[M+K]': '[M+K]+',
        '[M+NH4]': '[M+NH4]+',
        '[M+NH3]': '[M+NH3]+',
        '[M+NH5]': '[M+NH5]+',
        '[M+CH3OH+H]': '[M+CH3OH+H]+',
        '[M+CH3CN+H]': '[M+CH3CN+H]+',
        '[M+Na+CH3CN]': '[M+Na+CH3CN]+',
        '[M+HCOOH+H]': '[M+HCOOH+H]+',
        '[M+C2H6OS+H]': '[M+C2H6OS+H]+',
        '[M-H-H2O]': '[M-H-H2O]-',
        '[M+H-2H2O]': '[M+H-2H2O]+',
        '[M+H-3H2O]': '[M+H-3H2O]+',
        '[M+H-99]': '[M+H-99]+',
        '[M+H-H2]': '[M+H-H2]+',
        '[M+H-C9H10O5]': '[M+H-C9H10O5]+',
        '[M+H-C6H10O5]': '[M+H-C6H10O5]+',
        '[M+H-C3H8O]': '[M+H-C3H8O]+',
        '[M+H-CO]': '[M+H-CO]+',
        '[M-H-CO2-2HF]': '[M-H-CO2-2HF]-',
        '[M+K-2H]': '[M+K-2H]+',
        '[M+Na-2H]': '[M+Na-2H]+',
        '[M+2Na-H]': '[M+2Na-H]+',
        '[M+Na-H]': '[M+Na-H]+',
        '[2M+Na-2H]': '[2M+Na-2H]+',
        '[M+Cl-H]': '[M+Cl-H]-',
        '[M+]': '[M]+',
        '[M-]': '[M]-',
        '[M-H-H2O]': '[M-H-H2O]-',
    }
    
    for wrong, correct in fix_map.items():
        if normalized == wrong:
            normalized = correct
    
    if normalized.endswith(']') and not any(normalized.endswith(x) for x in ['+', '-', '2+', '2-', '3+', '3-']):
        normalized = normalized + charge if charge else normalized
    
    if normalized == '[M+Cl]-H]-':
        normalized = '[M+Cl-H]-'
    
    return normalized


def _get_mass_candidates(precursor_mz: float, adduct: str | None, charge: int | None) -> list[float]:
    candidates: list[float] = []

    adduct_info = None
    if adduct:
        normalized_adduct = normalize_adduct(adduct)
        adduct_info = adducts_look_up_dict.get(adduct) or adducts_look_up_dict.get(normalized_adduct)

    if adduct_info is not None:
        mass_shift = float(adduct_info["mass_shift"])
        n_mer = float(adduct_info["n_mer"])

        inferred_charge = charge if charge not in (None, 0) else adduct_info.get("charge")
        z = abs(int(inferred_charge)) if inferred_charge not in (None, 0) else 1

        if n_mer > 0:
            search_mass = (precursor_mz * z - mass_shift) / n_mer
            if math.isfinite(search_mass) and search_mass > 0:
                candidates.append(search_mass)

    proton_candidates: list[float] = []
    if charge is None:
        proton_candidates.extend([precursor_mz - PROTON_MASS, precursor_mz + PROTON_MASS])
    elif charge < 0:
        proton_candidates.append(precursor_mz + PROTON_MASS)
    else:
        proton_candidates.append(precursor_mz - PROTON_MASS)

    for proton_based in proton_candidates:
        if proton_based > 0:
            candidates.append(proton_based)

    candidates.append(precursor_mz)

    deduped: list[float] = []
    seen: set[float] = set()
    for value in candidates:
        rounded = round(value, 6)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(value)

    return deduped