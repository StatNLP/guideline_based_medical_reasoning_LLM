import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np

import random
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Normalize data and save. 
ts, oc, train_ind, valid_ind, test_ind = pickle.load(open('mimic_iii_preprocessed_extended_notes_icds.pkl', 'rb'))

stuff_to_use = {
    # Shared
    "GCS_eye", "GCS_motor", "GCS_verbal", "Bilirubin (Total)",
    "Urine", "FiO2", "PO2", "SBP",
    # SAPS-II only
    "HR", "Temperature", "BUN",
    "Sodium", "Potassium", "Bicarbonate", "WBC",
}
random.seed(42)
  
def get_sapsii(data_pat, pat, x_ts=24, pat_age=0):
    collection_dict = dict()
    data_var = data_pat[(data_pat['hour'] < x_ts) & (data_pat['hour'] >= x_ts-24)]
    data_var = data_var[data_var["value"] != -100.0]
    data_var['value2'] = data_var['value'] * data_var['std'] + data_var['mean']

    # --- Age ---
    # Age needs to come from the outcome/static data, passed in as pat_age
    # If not available, default to 0 points
    age = pat_age
    if age >= 80: age_saps = 18
    elif age >= 75: age_saps = 16
    elif age >= 70: age_saps = 15
    elif age >= 60: age_saps = 12
    elif age >= 40: age_saps = 7
    else: age_saps = 0
    collection_dict["Age_worst"] = age

    # --- HR (worst of min or max) ---
    min_hr = min(data_var[data_var['variable'] == 'HR']['value2'], default=80)
    max_hr = max(data_var[data_var['variable'] == 'HR']['value2'], default=80)
    def hr_score(hr):
        if hr < 40: return 11
        if hr >= 160: return 7
        if 120 <= hr < 160: return 4
        if 40 <= hr < 70: return 2
        return 0
    if hr_score(min_hr) >= hr_score(max_hr):
        hr_saps, hr_worst = hr_score(min_hr), min_hr
    else:
        hr_saps, hr_worst = hr_score(max_hr), max_hr
    collection_dict["HR_worst"] = hr_worst

    # --- SBP (worst of min or max) ---
    min_sbp = min(data_var[data_var['variable'] == 'SBP']['value2'], default=120)
    max_sbp = max(data_var[data_var['variable'] == 'SBP']['value2'], default=120)
    def sbp_score(sbp):
        if sbp < 70: return 13
        if 70 <= sbp < 100: return 5
        if sbp > 199: return 2
        return 0
    if sbp_score(min_sbp) >= sbp_score(max_sbp):
        sbp_saps, sbp_worst = sbp_score(min_sbp), min_sbp
    else:
        sbp_saps, sbp_worst = sbp_score(max_sbp), max_sbp
    collection_dict["SBP_worst"] = sbp_worst

    # --- Temperature (max) ---
    temp_worst = max(data_var[data_var['variable'] == 'Temperature']['value2'], default=37.0)
    if temp_worst >= 39: temp_saps = 3
    else: temp_saps = 0
    collection_dict["Temperature_worst"] = temp_worst

    # --- GCS (min) ---
    min_eye = min(data_var[data_var['variable'] == 'GCS_eye']['value2'], default=5)
    min_motor = min(data_var[data_var['variable'] == 'GCS_motor']['value2'], default=5)
    min_verbal = min(data_var[data_var['variable'] == 'GCS_verbal']['value2'], default=5)
    gcs_worst = min_eye + min_motor + min_verbal
    if gcs_worst >= 14: gcs_saps = 0
    elif gcs_worst >= 11: gcs_saps = 5
    elif gcs_worst >= 9: gcs_saps = 7
    elif gcs_worst >= 6: gcs_saps = 13
    else: gcs_saps = 26
    collection_dict["GCS_worst"] = gcs_worst

    # --- PaO2/FiO2 (min ratio) ---
    fio2 = min(data_var[data_var['variable'] == 'FiO2']['value2'], default=1)
    po2 = min(data_var[data_var['variable'] == 'PO2']['value2'], default=770)
    pf_worst = po2 / fio2
    if pf_worst < 100: pf_saps = 11
    elif pf_worst < 200: pf_saps = 9
    elif pf_worst >= 200: pf_saps = 6
    collection_dict["PaO2FiO2_worst"] = pf_worst

    # --- BUN (max) ---
    bun_worst = max(data_var[data_var['variable'] == 'BUN']['value2'], default=15)
    if bun_worst > 83: bun_saps = 10
    elif bun_worst >= 28: bun_saps = 6
    else: bun_saps = 0
    collection_dict["BUN_worst"] = bun_worst

    # --- Urine (sum over 24h) ---
    urine_worst = sum(data_var[data_var['variable'] == 'Urine']['value2'])
    if urine_worst < 500: urine_saps = 11
    elif urine_worst < 1000: urine_saps = 4
    else: urine_saps = 0
    collection_dict["Urine_worst"] = urine_worst

    # --- Sodium (worst of min or max) ---
    min_na = min(data_var[data_var['variable'] == 'Sodium']['value2'], default=138)
    max_na = max(data_var[data_var['variable'] == 'Sodium']['value2'], default=138)
    def na_score(na):
        if na < 125: return 5
        if na >= 145: return 1
        return 0
    if na_score(min_na) >= na_score(max_na):
        na_saps, na_worst = na_score(min_na), min_na
    else:
        na_saps, na_worst = na_score(max_na), max_na
    collection_dict["Sodium_worst"] = na_worst

    # --- Potassium (worst of min or max) ---
    min_k = min(data_var[data_var['variable'] == 'Potassium']['value2'], default=4.0)
    max_k = max(data_var[data_var['variable'] == 'Potassium']['value2'], default=4.0)
    def k_score(k):
        if k < 3.0 or k >= 5.0: return 5
        return 0
    if k_score(min_k) >= k_score(max_k):
        k_saps, k_worst = k_score(min_k), min_k
    else:
        k_saps, k_worst = k_score(max_k), max_k
    collection_dict["Potassium_worst"] = k_worst

    # --- Bicarbonate (min) ---
    bicarb_worst = min(data_var[data_var['variable'] == 'Bicarbonate']['value2'], default=22)
    if bicarb_worst < 15: bicarb_saps = 6
    elif bicarb_worst < 20: bicarb_saps = 3
    else: bicarb_saps = 0
    collection_dict["Bicarbonate_worst"] = bicarb_worst

    # --- WBC (worst of min or max) ---
    min_wbc = min(data_var[data_var['variable'] == 'WBC']['value2'], default=10)
    max_wbc = max(data_var[data_var['variable'] == 'WBC']['value2'], default=10)
    def wbc_score(wbc):
        if wbc < 1.0: return 12
        if wbc >= 20: return 3
        return 0
    if wbc_score(min_wbc) >= wbc_score(max_wbc):
        wbc_saps, wbc_worst = wbc_score(min_wbc), min_wbc
    else:
        wbc_saps, wbc_worst = wbc_score(max_wbc), max_wbc
    collection_dict["WBC_worst"] = wbc_worst

    # --- Bilirubin (max) ---
    bilir_worst = max(data_var[data_var['variable'] == 'Bilirubin (Total)']['value2'], default=1)
    if bilir_worst >= 6: bilir_saps = 9
    elif bilir_worst >= 4: bilir_saps = 4
    else: bilir_saps = 0
    collection_dict["Bilirubin_worst"] = bilir_worst

    # --- Total ---
    sapsii_components = (age_saps, hr_saps, sbp_saps, temp_saps, gcs_saps, pf_saps,
                         bun_saps, urine_saps, na_saps, k_saps, bicarb_saps,
                         wbc_saps, bilir_saps)
    total_sapsii = sum(sapsii_components)

    # Logit and probability of hospital mortality (Le Gall et al., 1993)
    import math
    logit = -7.7631 + 0.0737 * total_sapsii + 0.9971 * math.log(total_sapsii + 1)
    probability = math.exp(logit) / (1 + math.exp(logit))
    collection_dict["sapsii_logit"] = logit
    collection_dict["sapsii_probability"] = probability

    sapsii_vars = {"HR", "SBP", "Temperature", "GCS_eye", "GCS_motor", "GCS_verbal",
                   "FiO2", "PO2", "BUN", "Urine",
                   "Sodium", "Potassium", "Bicarbonate", "WBC", "Bilirubin (Total)"}
    valid_measurements = data_var[data_var.variable.isin(sapsii_vars)]

    return sapsii_components, total_sapsii, collection_dict, len(valid_measurements)


def generate_synthetic_reasoning_sapsii(sapsii_components, total_sapsii, collection_dict):
    collection_dict = {x: round(y, 3) for x, y in collection_dict.items()}
    text = "What is the SAPS-II score of the patient?\n"
    text += "We need to calculate the SAPS-II scores given the extracted values.\n"
    text += f"The patient's age is {collection_dict['Age_worst']}, contributing {sapsii_components[0]} points.\n"
    text += f"The worst HR value is {collection_dict['HR_worst']}, contributing {sapsii_components[1]} points.\n"
    text += f"The worst SBP value is {collection_dict['SBP_worst']}, contributing {sapsii_components[2]} points.\n"
    text += f"The worst Temperature value is {collection_dict['Temperature_worst']}, contributing {sapsii_components[3]} points.\n"
    text += f"The worst GCS value is {collection_dict['GCS_worst']}, contributing {sapsii_components[4]} points.\n"
    text += f"The worst PaO2/FiO2 ratio is {collection_dict['PaO2FiO2_worst']}, contributing {sapsii_components[5]} points.\n"
    text += f"The worst BUN value is {collection_dict['BUN_worst']}, contributing {sapsii_components[6]} points.\n"
    text += f"The total Urine output is {collection_dict['Urine_worst']}, contributing {sapsii_components[7]} points.\n"
    text += f"The worst Sodium value is {collection_dict['Sodium_worst']}, contributing {sapsii_components[8]} points.\n"
    text += f"The worst Potassium value is {collection_dict['Potassium_worst']}, contributing {sapsii_components[9]} points.\n"
    text += f"The worst Bicarbonate value is {collection_dict['Bicarbonate_worst']}, contributing {sapsii_components[10]} points.\n"
    text += f"The worst WBC value is {collection_dict['WBC_worst']}, contributing {sapsii_components[11]} points.\n"
    text += f"The worst Bilirubin value is {collection_dict['Bilirubin_worst']}, contributing {sapsii_components[12]} points.\n"
    text += f"To summarize: the patient has a total SAPS-II score of {total_sapsii}.\n"
    text += f"The logit is computed as -7.7631 + 0.0737 * {total_sapsii} + 0.9971 * ln({total_sapsii} + 1) = {collection_dict['sapsii_logit']}.\n"
    text += f"The predicted hospital mortality probability is e^logit / (1 + e^logit) = {collection_dict['sapsii_probability']}.\n"
    return text

var_to_ind= {'ALP': 1, 'ALT': 2, 'AST': 3, 'Albumin': 4, 'Albumin 25%': 5, 'Albumin 5%': 6,
              'Amiodarone': 7, 'Anion Gap': 8, 'Antibiotics': 9, 'BUN': 10, 'Base Excess': 11,
              'Basophils': 12, 'Bicarbonate': 13, 'Bilirubin (Direct)': 14, 'Bilirubin (Indirect)': 15,
              'Bilirubin (Total)': 16, 'CRR': 17, 'Calcium Free': 18, 'Calcium Gluconate': 19,
              'Calcium Total': 20, 'Cefazolin': 21, 'Chest Tube': 22, 'Chloride': 23, 'Colloid': 24,
              'Creatinine Blood': 25, 'Creatinine Urine': 26, 'D5W': 27, 'DBP': 28, 'Dextrose Other': 29,
              'Dobutamine': 30, 'Dopamine': 31, 'EBL': 32, 'Emesis': 33, 'Eoisinophils': 34,
              'Epinephrine': 35, 'Famotidine': 36, 'Fentanyl': 37, 'FiO2': 38, 'Fiber': 39,
              'Free Water': 40, 'Fresh Frozen Plasma': 41, 'Furosemide': 42, 'GCS_eye': 43,
              'GCS_motor': 44, 'GCS_verbal': 45, 'GT Flush': 46, 'Gastric': 47, 'Gastric Meds': 48,
              'Glucose (Blood)': 49, 'Glucose (Serum)': 50, 'Glucose (Whole Blood)': 51,
              'HR': 52, 'Half Normal Saline': 53, 'Hct': 54, 'Heparin': 55, 'Hgb': 56,
              'Hydralazine': 57, 'Hydromorphone': 58, 'INR': 59, 'Insulin Humalog': 60,
              'Insulin NPH': 61, 'Insulin Regular': 62, 'Insulin largine': 63,
              'Intubated': 64, 'Jackson-Pratt': 65, 'KCl': 66, 'KCl (Bolus)': 67,
              'LDH': 68, 'Lactate': 69, 'Lactated Ringers': 70, 'Levofloxacin': 71,
              'Lorazepam': 72, 'Lymphocytes': 73, 'Lymphocytes (Absolute)': 74,
              'MBP': 75, 'MCH': 76, 'MCHC': 77, 'MCV': 78, 'Magnesium': 79,
              'Magnesium Sulfate (Bolus)': 80,  'Magnesium Sulphate': 81,
              'Mechanically ventilated': 82, 'Metoprolol': 83, 'Midazolam': 84,
              'Milrinone': 85, 'Monocytes': 86, 'Morphine Sulfate': 87,
              'Neosynephrine': 88, 'Neutrophils': 89, 'Nitroglycerine': 90,
              'Nitroprusside': 91, 'Norepinephrine': 92, 'Normal Saline': 93,
              'O2 Saturation': 94, 'OR/PACU Crystalloid': 95, 'PCO2': 96,
              'PO intake': 97, 'PO2': 98, 'PT': 99, 'PTT': 100, 'Packed RBC': 101,
              'Pantoprazole': 102, 'Phosphate': 103, 'Piggyback': 104, 'Piperacillin': 105,
              'Platelet Count': 106, 'Potassium': 107, 'Pre-admission Intake': 108,
              'Pre-admission Output': 109, 'Propofol': 110, 'RBC': 111, 'RDW': 112,
              'RR': 113, 'Residual': 114, 'SBP': 115, 'SG Urine': 116, 'Sodium': 117,
              'Solution': 118, 'Sterile Water': 119, 'Stool': 120, 'TPN': 121,
              'Temperature': 122, 'Total CO2': 123, 'Ultrafiltrate': 124, 'Urine': 125,
              'Vancomycin': 126, 'Vasopressin': 127, 'WBC': 128, 'Weight': 129,
              'pH Blood': 130, 'pH Urine': 131}
import math
import time as timer

new_information = []
#for ts_ind in ts.ts_ind.unique():
#for ts_ind in [0.0]:
import tqdm 



for ts_ind in tqdm.tqdm(ts.ts_ind.unique(), total=ts.ts_ind.unique().shape[0]):
    #for ts_ind in [0.0]:
    ts_pat = ts[ts["ts_ind"]==ts_ind]
    ts_pat = ts_pat[ts_pat["value"]!=-100.0]
    age_row = ts_pat[ts_pat["variable"]=="Age"]
    age = ts_pat[ts_pat["variable"]=="Age"]["value"].iloc[0]
    for time in [24, 48, 72, 96, 120]:

        sapsii_components, total_sapsii, collection_dict, len_valid_measurements = get_sapsii(ts_pat, ts_ind, time, age)
        if len_valid_measurements == 0:
            break
        time = time-0.0000001
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "total-sofa", "value": sum(sofa), "TABLE": "analysis", "mean": 0, "std": 1})
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "cns-sofa", "value": sofa[0], "TABLE": "analysis", "mean": 0, "std": 1})
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "cardiovascular-sofa", "value": sofa[1], "TABLE": "analysis", "mean": 0, "std": 1})
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "respiratory-sofa", "value": sofa[2], "TABLE": "analysis", "mean": 0, "std": 1})
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "coagulation-sofa", "value": sofa[3], "TABLE": "analysis", "mean": 0, "std": 1})
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "liver-sofa", "value": sofa[4], "TABLE": "analysis", "mean": 0, "std": 1})
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "renal-sofa", "value": sofa[5], "TABLE": "analysis", "mean": 0, "std": 1})


        if True:
            text = generate_synthetic_reasoning_sapsii(sapsii_components, total_sapsii, collection_dict)
            new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "synthetic_sapsii_label", "TABLE": "TEXT", "textvalue": text})

new_df = pd.DataFrame(new_information)
ts = pd.concat([ts, new_df]) 


ts.sort_values(by=["ts_ind", "hour"], inplace=True)


data = ts[ts["ts_ind"]==0.0]
data.to_csv("for_stefan_patient0_sapsii.csv")
    
pickle.dump([ts, oc, train_ind, valid_ind, test_ind], open('mimic_iii_preprocessed_sapsii.pkl', 'wb'))



