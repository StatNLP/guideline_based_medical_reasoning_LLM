import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import random
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Normalize data and save. 
ts, oc, train_ind, valid_ind, test_ind = pickle.load(open('mimic_iii_preprocessed_extended_notes_icds.pkl', 'rb'))
print(ts["variable"].unique())
data_icd = (ts[ts["variable"]=="ICD9"].value.unique())

blub=(ts[ts["variable"]=="rrt"])
print(len(blub))
#print(blub)

MIMIC_ICD9_PATTERNS = {
    # CKD split into 3 classes based on severity
    # CKD2: N18.3, N18.30, N18.31, N18.32, N18.9 -> ICD9: 585.3, 585.9
    "ckd2": ['5853', '5859'],
    
    # CKD3: N18.4 -> ICD9: 585.4
    "ckd3": ['5854'],
    
    # CKD4: N18.5, I12.0, I13.11, I13.2, N18.6 -> ICD9: 585.5, 585.6, 403.01, 403.11, 403.91, 404.02, 404.03, 404.13, 404.93
    "ckd4": ['5855', '5856', '40301', '40311', '40391', '40402', '40403', '40413', '40493'],
    
    # Combined CKD (all stages) for backwards compatibility
    "ckd": ['5853', '5859', '5854', '5855', '5856', '40301', '40402', '40403', '40311', '40391', '40413', '40493'],
    
    "leukemia_lymphoma": ['2010', '2011', '2016', '2017', '2012', '2019', '2020', '2028', '2007', '2001', '2002', '2021', '2022', '2027', '2006', '2003', '2038', '2030', '2031', '2386', '2040', '2041', '2048', '2024', '2049', '2050', '2051', '2053', '2059', '2060', '2061', '2062', '2069', '2070', '2072', '2078', '2080', '2081', '2089', '2025', '2026', '2029', '2000', '2009', '2014', '2015', '20300', '20301', '20302', '20310', '20311', '20312', '20400', '20401', '20402', '20410', '20411', '20412', '20500', '20501', '20502', '20510', '20511', '20512', '20600', '20601', '20610', '20611', '20700', '20701', '20720', '20721', '20800', '20801', '20810', '20811', '20890', '20891'],
    
    "liver": ['4560', '4561', '45621', '5728', '5723', '5724'],
    
    #"renal_dialysis": ['40301', '40412', '5851', '5852', '5853', '5854', '5855', '5856', '5859', '586', '5880', 'V561', 'V562', 'V560', 'V568', 'V420', 'V4511', '40311', '40391', '40402', '40492', 'V4512', 'V5631', 'V5632'],
    
    "dementia": ['29011', '2900', '29041', '29040', '29043', '29410', '29411', '2903', '3312', '29042', '2908', '2909', '29010', '29012', '29013', '29020', '29021'],
}

def categorize_icds(data_icd):
    used_codes = set()
    used_clas = set()
    for icd_code in data_icd:
        icd_code = str(icd_code)
        for clas, codes in MIMIC_ICD9_PATTERNS.items():
            for code in codes:
                if icd_code.startswith(code):
                    used_clas.add(clas)
                    used_codes.add(icd_code)
    used_codes = list(used_codes)
    used_clas = list(used_clas)
    return used_codes, used_clas

stuff_to_use = {"GCS_eye", "GCS_motor", "GCS_verbal", "Bilirubin (Total)", "Platelet Count", "Urine", "Creatinine Blood", "FiO2", "PO2", "Weight", "Dopamine", "Dobutamine", "Epinephrine", "Norepinephrine", "SBP", "DBP"}

random.seed(42)
  
  
def get_sofa(data_pat, pat, x_ts=24, used_codes=None, used_clas=None, rrt_used=False): #patient number, x_ts for hour
    collection_dict = dict()
    data_var = data_pat[(data_pat['hour'] < x_ts) & (data_pat['hour'] >= x_ts-24)]  # values of last 24 hours
    data_var=data_var[data_var["value"]!=-100.0]
    data_var['value2'] = data_var['value']*data_var['std']+data_var['mean']  # get real values
    # GCS: min_eye, min_motor, min_verbal = 5, 5, 5
    min_eye = min(data_var[data_var['variable']=='GCS_eye']['value2'], default=5)
    min_motor = min(data_var[data_var['variable']=='GCS_motor']['value2'], default=5)
    min_verbal = min(data_var[data_var['variable']=='GCS_verbal']['value2'], default=5)
    GCS = min_eye + min_motor + min_verbal
    if GCS > 14: GCS_sofa = 0
    elif GCS > 12: GCS_sofa = 1
    elif GCS > 9:  GCS_sofa = 2
    elif GCS > 5:  GCS_sofa = 3
    else: GCS_sofa = 4
    if "dementia" in used_clas:
        GCS_sofa=2
    collection_dict["minimum GCS_eye"] = min_eye
    collection_dict["minimum GCS_motor"] = min_motor
    collection_dict["minimum GCS_verbal"] = min_verbal
    collection_dict["GCS"] = GCS

    if "dementia" in used_clas:
        GCS_sofa=2
    # print('GCS_sofa is', GCS_sofa, ';     GCS is', GCS,'; GCS eye', min_eye, '; GCS motor', min_motor, '; GCS verbal', min_verbal)

    # bilirubin Liver SOFA
    bilir = max(data_var[data_var['variable']=='Bilirubin (Total)'] ['value2'], default=1)
    if bilir > 12: bilir_sofa = 4
    elif bilir > 6: bilir_sofa = 3
    elif bilir > 2: bilir_sofa = 2
    elif bilir > 1.2: bilir_sofa = 1
    else: bilir_sofa = 0
    collection_dict["maximum Bilirubin (Total)"] = bilir
    # print('bilir_Sofa is', bilir_sofa, ';   bilirubin is', bilir)
    if "liver" in used_clas:
        bilir_sofa=2
    # Coagulation (Platelets)
    plate = min(data_var[data_var['variable']=='Platelet Count'] ['value2'], default=160)
    if plate > 150: plate_sofa = 0
    elif plate > 100: plate_sofa = 1
    elif plate > 50: plate_sofa = 2
    elif plate > 20: plate_sofa = 4
    else: plate_sofa = 4
    collection_dict["minimum Platelet Count"] = plate
    # print('plate_sofa is', plate_sofa, ';   platelet count is', plate)
    if "leukemia_lymphoma" in used_clas:
        plate_sofa=4
    # print('Urinmenge 24h', sum(data_var[data_var['variable']=='Urine']['value2']))
    urine = sum(data_var[data_var['variable']=='Urine']['value2'])
    creat = max(data_var[data_var['variable']=='Creatinine Blood']['value2'], default=1)
    collection_dict["sum of Urine"] = urine
    collection_dict["maximum Creatinine Blood"] = creat
    if (urine < 200) or (creat > 5): renal_sofa = 4
    elif  (urine < 500) or (creat > 3.5): renal_sofa = 3
    elif creat > 2.0: renal_sofa = 2
    elif creat > 1.2: renal_sofa = 1
    else: renal_sofa = 0
    # print('renal_sofa:',renal_sofa,';       urine 24:',urine,'; creat:', creat)
    if used_rrt:
        renal_sofa=4
    elif "cdk4" in used_clas:
        renal_sofa=4
    elif "cdk3" in used_clas:
        renal_sofa=3
    elif "cdk2" in used_clas:
        renal_sofa=2
    *CS_data, collection_dict = get_CS(data_pat, x_ts, collection_dict)
    cs_sofa = CS_SOFA(CS_data)

    fio2 = min(data_var[data_var['variable']=='FiO2'] ['value2'], default=1)
    po2 = min(data_var[data_var['variable']=='PO2'] ['value2'], default=770)
    pao2fio2 = po2/fio2
    collection_dict["minimum FiO2"]=fio2
    collection_dict["minimum PO2"]=po2
    collection_dict["pao2fio2"]=pao2fio2
    if pao2fio2<100: resp=4
    elif pao2fio2<200: resp=3
    elif pao2fio2<300:resp=2
    elif pao2fio2<400:resp=1
    else: resp=0
    valid_measurements = (data_var[data_var.variable.isin(stuff_to_use)][["ts_ind", "hour", "variable", "value2"]])
    valid_measurements = (data_var[data_var.variable.isin(stuff_to_use)][["ts_ind", "hour", "variable", "value2"]])
    return GCS_sofa, cs_sofa, resp, plate_sofa, bilir_sofa, renal_sofa, collection_dict, len(valid_measurements)


def get_CS(data_pat, x_ts, collection_dict):
    data_var = data_pat[data_pat['variable'].isin(['Dobutamine','Dopamine','Epinephrine','Norepinephrine','Weight'])]
    data_var['value2'] = data_var['value']*data_var['std']+data_var['mean']


    data_var = data_var[(data_var['hour'] < x_ts) & (data_var['hour'] > x_ts-24)]

    weight = min(data_var[data_var['variable']=='Weight']['value2'], default=80)  # set default weight to 80kg.

    data_dop = data_var[data_var['variable']=='Dopamine']
    if data_dop.empty==0:
        data_dop['length']=-data_dop['hour'].diff(periods=-1)
        data_dop['length'].iloc[-1] = 3  # set by hand.
        denominator = data_dop['length'] * 60 * weight
        try:
            data_dop['dop_rate'] = np.divide(
                data_dop['value2'] * 1000,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=np.abs(denominator) > 0.00001
            )
            max_dop = max(data_dop['dop_rate'], default=0)

        except: max_dop = 0
    else: max_dop = 0

    data_dobu = data_var[data_var['variable']=='Dobutamine']
    if data_dobu.empty==0:
        data_dobu['length']=-data_dobu['hour'].diff(periods=-1)
        data_dobu['length'].iloc[-1] = 3  # set by hand.
        denominator = data_dobu['length'] * 60 * weight
        try:
            data_dobu['dop_rate'] = np.divide(
            data_dobu['value2'] * 1000,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=np.abs(denominator) > 0.00001
            )
            max_dobu = max(data_dobu['dop_rate'], default=0)
        except: max_dobu = 0
    else: max_dobu = 0

    data_epi = data_var[data_var['variable']=='Epinephrine']
    if data_epi.empty==0:
        data_epi['length']=-data_epi['hour'].diff(periods=-1)
        data_epi['length'].iloc[-1] = 3  # set by hand.
        denominator = data_epi['length'] * 60 * weight
        try:
            data_epi['dop_rate'] = np.divide(
                data_epi['value2'] * 1000,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=np.abs(denominator) > 0.00001
            )
            max_epi = max(data_epi['dop_rate'], default=0)
        except: max_epi = 0
    else: max_epi = 0

    data_nore = data_var[data_var['variable']=='Norepinephrine']
    if data_nore.empty==0:
        data_nore['length']=-data_nore['hour'].diff(periods=-1)
        data_nore['length'].iloc[-1] = 3  # set by hand.
        denominator = data_nore['length'] * 60 * weight
        try:
            data_nore['dop_rate'] = np.divide(
                data_nore['value2'] * 1000,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=np.abs(denominator) > 0.00001
            )
            max_nore = max(data_nore['dop_rate'], default=0)
        except: max_nore = 0
    else: max_nore = 0

    data_pat['value2'] = data_pat['value']*data_pat['std']+data_pat['mean']  # get real values
    data_var = data_pat[(data_pat['hour'] < x_ts) & (data_pat['hour'] > x_ts-24)]

    sbp_data = data_var[data_var['variable'].isin(['SBP'])]
    dbp_data = data_var[data_var['variable'].isin(['DBP'])]

    map_data = pd.concat([dbp_data.set_index('hour'),sbp_data.set_index('hour')], axis=1, join='inner')
    map_data = map_data['value2']
    map_data.columns = ['DBP','SBP']
    map_data['MAP'] = 2/3*map_data['DBP'] +1/3 *map_data['SBP']
    map_data = map_data[map_data['MAP']>0]
    # print( map_data)
    min_map = min(map_data['MAP'], default= 100)
    collection_dict["weight"]=weight
    collection_dict["minimum MAP"] = min_map
    collection_dict["max_dop"] = max_dop
    collection_dict["max_dobu"] = max_dobu
    collection_dict["max_epi"] = max_epi
    collection_dict["max_nore"] = max_nore
    return min(map_data['MAP'], default= 100), max_dop, max_dobu, max_epi, max_nore, collection_dict

def CS_SOFA(data):
    map = data[0]
    dop, dobu, epi, nore = data[1:5]
    # print('CS data: mdden', data)
    if (dop > 15) or (epi > 0.1) or (nore > 0.01): CS = 4
    elif (dop > 5) or (epi > 0) or (nore > 0): CS = 3
    elif (dop > 0) or (dobu > 0): CS = 2
    elif map < 70: CS = 1
    else: CS = 0
    # print('CS Sofa is:', CS)
    return CS



def get_aug_sofa(data_pat, pat, x_ts=24, old_dict=dict(), used_codes=None, used_clas=None, used_rrt=False ): #patient number, x_ts for hour
    collection_dict = dict()
    data_var = data_pat[(data_pat['hour'] < x_ts) & (data_pat['hour'] >= x_ts-24)]  # values of last 24 hours
    data_var=data_var[data_var["value"]!=-100.0]
    data_var['value2'] = data_var['value']*data_var['std']+data_var['mean']  # get real values
    # GCS: min_eye, min_motor, min_verbal = 5, 5, 5
    min_eye = min(data_var[data_var['variable']=='GCS_eye']['value2'], default=5 if "minimum GCS_eye" not in old_dict else old_dict["minimum GCS_eye"])
    min_motor = min(data_var[data_var['variable']=='GCS_motor']['value2'], default=5 if "minimum GCS_motor" not in old_dict else old_dict["minimum GCS_motor"])
    min_verbal = min(data_var[data_var['variable']=='GCS_verbal']['value2'], default=5 if "minimum GCS_verbal" not in old_dict else old_dict["minimum GCS_verbal"])
    GCS = min_eye + min_motor + min_verbal
    if GCS > 14: GCS_sofa = 0
    elif GCS > 12: GCS_sofa = 1
    elif GCS > 9:  GCS_sofa = 2
    elif GCS > 5:  GCS_sofa = 3
    else: GCS_sofa = 4
    if "dementia" in used_clas:
        GCS_sofa=2
    collection_dict["minimum GCS_eye"] = min_eye
    collection_dict["minimum GCS_motor"] = min_motor
    collection_dict["minimum GCS_verbal"] = min_verbal
    collection_dict["GCS"] = GCS
    # print('GCS_sofa is', GCS_sofa, ';     GCS is', GCS,'; GCS eye', min_eye, '; GCS motor', min_motor, '; GCS verbal', min_verbal)

    # bilirubin Liver SOFA
    bilir = max(data_var[data_var['variable']=='Bilirubin (Total)'] ['value2'], default=1 if "maximum Bilirubin (Total)" not in old_dict else old_dict["maximum Bilirubin (Total)"])
    if bilir > 12: bilir_sofa = 4
    elif bilir > 6: bilir_sofa = 3
    elif bilir > 2: bilir_sofa = 2
    elif bilir > 1.2: bilir_sofa = 1
    else: bilir_sofa = 0
    collection_dict["maximum Bilirubin (Total)"] = bilir
    # print('bilir_Sofa is', bilir_sofa, ';   bilirubin is', bilir)

    if "liver" in used_clas:
        bilir_sofa=2
    # Coagulation (Platelets)
    plate = min(data_var[data_var['variable']=='Platelet Count'] ['value2'], default=160  if "minimum Platelet Count" not in old_dict else old_dict["minimum Platelet Count"])
    if plate > 150: plate_sofa = 0
    elif plate > 100: plate_sofa = 1
    elif plate > 50: plate_sofa = 2
    elif plate > 20: plate_sofa = 4
    else: plate_sofa = 4
    collection_dict["minimum Platelet Count"] = plate 
    # print('plate_sofa is', plate_sofa, ';   platelet count is', plate)

    if "leukemia_lymphoma" in used_clas:
        plate_sofa=4
    # print('Urinmenge 24h', sum(data_var[data_var['variable']=='Urine']['value2']))
    urine = sum(data_var[data_var['variable']=='Urine']['value2'])
    creat = max(data_var[data_var['variable']=='Creatinine Blood']['value2'], default=1 if "maximum Creatinine Blood" not in old_dict else old_dict["maximum Creatinine Blood"])
    collection_dict["sum of Urine"] = urine
    collection_dict["maximum Creatinine Blood"] = creat
    if (urine < 200) or (creat > 5): renal_sofa = 4
    elif  (urine < 500) or (creat > 3.5): renal_sofa = 3
    elif creat > 2.0: renal_sofa = 2
    elif creat > 1.2: renal_sofa = 1
    else: renal_sofa = 0
    # print('renal_sofa:',renal_sofa,';       urine 24:',urine,'; creat:', creat)
    if used_rrt:
        renal_sofa=4
    elif "cdk4" in used_clas:
        renal_sofa=4
    elif "cdk3" in used_clas:
        renal_sofa=3
    elif "cdk2" in used_clas:
        renal_sofa=2
    *CS_data, collection_dict = get_aug_CS(data_var, x_ts, collection_dict, old_dict)
    cs_sofa = aug_CS_SOFA(CS_data)

    fio2 = min(data_var[data_var['variable']=='FiO2'] ['value2'], default=1  if "minimum FiO2" not in old_dict else old_dict["minimum FiO2"])
    po2 = min(data_var[data_var['variable']=='PO2'] ['value2'], default=770  if "minimum PO2" not in old_dict else old_dict["minimum PO2"])
    pao2fio2 = po2/fio2
    collection_dict["minimum FiO2"]=fio2
    collection_dict["minimum PO2"]=po2
    collection_dict["pao2fio2"]=pao2fio2
    if pao2fio2<100: resp=4
    elif pao2fio2<200: resp=3
    elif pao2fio2<300:resp=2
    elif pao2fio2<400:resp=1
    else: resp=0
    valid_measurements = (data_var[data_var.variable.isin(stuff_to_use)][["ts_ind", "hour", "variable", "value2"]])
    valid_measurements = (data_var[data_var.variable.isin(stuff_to_use)][["ts_ind", "hour", "variable", "value2"]])
    return GCS_sofa, cs_sofa, resp, plate_sofa, bilir_sofa, renal_sofa, collection_dict, len(valid_measurements)

def get_aug_CS(data_pat, x_ts, collection_dict, old_dict):
    data_var = data_pat[data_pat['variable'].isin(['Dobutamine','Dopamine','Epinephrine','Norepinephrine','Weight'])]
    data_var['value2'] = data_var['value']*data_var['std']+data_var['mean']


    data_var = data_var[(data_var['hour'] < x_ts) & (data_var['hour'] > x_ts-24)]

    weight = min(data_var[data_var['variable']=='Weight']['value2'], default=80   if "weight" not in old_dict else old_dict["weight"])  # set default weight to 80kg.
    data_dop = data_var[data_var['variable']=='Dopamine']
    if data_dop.empty==0:
        data_dop['length']=-data_dop['hour'].diff(periods=-1)
        data_dop['length'].iloc[-1] = 3  # set by hand.
        #data_dop['dop_rate'] = data_dop['value2']/data_dop['length']/60/weight*1000
        denominator = data_dop['length'] * 60 * weight
        try:
            data_dop['dop_rate'] = np.divide(
                data_dop['value2'] * 1000,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=np.abs(denominator) > 0.00001
            )
            max_dop = max(data_dop['dop_rate'], default=0 if "max_dop" not in old_dict else old_dict["max_dop"])
        except: max_dop = 0
    else: max_dop = 0

    data_dobu = data_var[data_var['variable']=='Dobutamine']
    if data_dobu.empty==0:
        data_dobu['length']=-data_dobu['hour'].diff(periods=-1)
        data_dobu['length'].iloc[-1] = 3  # set by hand.

        denominator = data_dobu['length'] * 60 * weight
        try:
            data_dobu['dop_rate'] = np.divide(
                data_dobu['value2'] * 1000,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=np.abs(denominator) > 0.00001
            )

            #data_dobu['dop_rate'] = data_dobu['value2']/data_dobu['length']/60/weight*1000
            max_dobu = max(data_dobu['dop_rate'], default=0   if "max_dobu" not in old_dict else old_dict["max_dobu"])
        except: max_dobu = 0
    else: max_dobu = 0

    data_epi = data_var[data_var['variable']=='Epinephrine']
    if data_epi.empty==0:
        data_epi['length']=-data_epi['hour'].diff(periods=-1)
        data_epi['length'].iloc[-1] = 3  # set by hand.

        denominator = data_epi['length'] * 60 * weight
        try:
            data_epi['dop_rate'] = np.divide(
                data_epi['value2'] * 1000,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=np.abs(denominator) > 0.00001
            )

            #data_epi['dop_rate'] = data_epi['value2']/data_epi['length']/60/weight*1000
            max_epi = max(data_epi['dop_rate'], default=0   if "max_epi" not in old_dict else old_dict["max_epi"])
        except: max_epi = 0
    else: max_epi = 0

    data_nore = data_var[data_var['variable']=='Norepinephrine']
    if data_nore.empty==0:
        data_nore['length']=-data_nore['hour'].diff(periods=-1)
        data_nore['length'].iloc[-1] = 3  # set by hand
        
        try:
            #data_nore['dop_rate'] = data_nore['value2']/data_nore['length']/60/weight*1000
            denominator = data_nore['length'] * 60 * weight
            #data_nore['dop_rate'] = np.where(
            #    denominator == 0,
            #    0,
            #    data_nore['value2'] / denominator * 1000
            #)
            data_nore['dop_rate'] = np.divide(
                data_nore['value2'] * 1000,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=np.abs(denominator) > 0.00001
            )
            max_nore = max(data_nore['dop_rate'], default=0   if "max_nore" not in old_dict else old_dict["max_nore"])

        except:
            max_nore=0
    else: max_nore = 0
    data_pat['value2'] = data_pat['value']*data_pat['std']+data_pat['mean']  # get real values
    data_var = data_pat[(data_pat['hour'] < x_ts) & (data_pat['hour'] > x_ts-24)]

    sbp_data = data_var[data_var['variable'].isin(['SBP'])]
    dbp_data = data_var[data_var['variable'].isin(['DBP'])]
    map_data = pd.concat([dbp_data.set_index('hour'),sbp_data.set_index('hour')], axis=1, join='inner')
    map_data = map_data['value2']
    map_data.columns = ['DBP','SBP']
    map_data['MAP'] = 2/3*map_data['DBP'] +1/3 *map_data['SBP']
    map_data = map_data[map_data['MAP']>0]
    # print( map_data)
    min_map = min(map_data['MAP'], default= 100 if "minimum MAP" not in old_dict else old_dict["minimum MAP"])
    collection_dict["weight"]=weight
    collection_dict["minimum MAP"] = min_map
    collection_dict["max_dop"] = max_dop
    collection_dict["max_dobu"] = max_dobu
    collection_dict["max_epi"] = max_epi
    collection_dict["max_nore"] = max_nore
    return min(map_data['MAP'], default= 100), max_dop, max_dobu, max_epi, max_nore, collection_dict

def aug_CS_SOFA(data):
    map = data[0]
    dop, dobu, epi, nore = data[1:5]
    # print('CS data: mdden', data)
    if (dop > 15) or (epi > 0.1) or (nore > 0.01): CS = 4
    elif (dop > 5) or (epi > 0) or (nore > 0): CS = 3
    elif (dop > 0) or (dobu > 0): CS = 2
    elif map < 70: CS = 1
    else: CS = 0
    # print('CS Sofa is:', CS)
    return CS


def susinfo(data_pat, pat, x_ts):
    datain = data_pat[(data_pat['hour'] < x_ts) & (data_pat['hour'] >= x_ts-24)] 
    AB = datain[datain['variable']=='Antibiotics']  # 1 490 822
    BC = datain[datain['variable']=='Blood Culture'] # 57 270
    if AB.empty and BC.empty:
        return 0
    else:
        return 1
def generate_synthetic_analysis(data_pat, pat, x_ts, sofa, susinf):
    data_var = data_pat[(data_pat['hour'] < x_ts) & (data_pat['hour'] >= x_ts-24)]
    data_var = data_var[data_pat["variable"]!="Age"]
    data_var = data_var[data_pat["variable"]!="Gender"]
    data_var['value2'] = data_var['value']*data_var['std']+data_var['mean']
    blub = data_var.groupby("variable").agg({"value2": ["min", "mean", "max", "count"]})
    texts = []
    for i in range(10):
        try:
            variables = blub.sample(n=random.randint(1, 10))
        except: continue
        text = ""
        for variable in variables.iterrows():
            aggregator =  random.choice(["min", "mean", "max"])
            if aggregator == "max":
                word = "highest"
            if aggregator == "mean":
                word = "average"
            if aggregator == "min":
                word = "lowest"

            text += "What is the " + word + " " + variable[0] +"?" + " The " + word + " " + variable[0] + " is " + str(round(variable[1]["value2"][aggregator], 3)) + ".\n"
        texts.append(text)
    return texts
import time as timer

def generate_synthetic_sofa(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf, susinf_collector, septic):
    collection_dict = {x:round(y, 3) for x,y in collection_dict.items()}
    text = ""
    #if susinf:
    #    text+= "The doctors suspect an infection, which is why a Blood Culture was taken and antibiotics were given in the last 24 hours."
    #else:
    #    text+= "The doctors did not suspect an infection in the last 24 hours because either no antibitotics were given or no blood culture was taken."
    if True:
        text+= " The CNS sofa is {}, this is because the values of GCS_eye ({}), GCS_motor ({}) and GCS_verbal ({}) produces the sum {}.".format(sofa[0], collection_dict["GCS_eye"], collection_dict["GCS_motor"], collection_dict["GCS_verbal"], collection_dict["GCS"])
    if True:
        text+= " The cardiovascular sofa is {}, because minimum MAP is {},  max Dopamine is {}, max Dobutamine is {}, max Epinephrine is {} and max Norepinephrine is {} with a patient weight of {} kg.".format(sofa[1], collection_dict["MAP"], collection_dict["max_dop"], collection_dict["max_dobu"], collection_dict["max_epi"], collection_dict["max_nore"], collection_dict["weight"])
    if True:
        text+= " The respiratory sofa is {}, because the calculated PAO2FiO2 is {} given PO2 ({}) and FiO2 ({}).".format(sofa[2], collection_dict["pao2fio2"], collection_dict["PO2"], collection_dict["FiO2"])
    if True:
        text+= " The coagulation sofa is {}, because the Platelet count is {}.".format(sofa[3], collection_dict["Platelet Count"])
    if True:
        text+= " The liver sofa is {}, because the Bilirubin (Total) is {}.".format(sofa[4], collection_dict["Bilirubin (Total)"])
    if True:
        text+= " The renal sofa is {}, because Urine output is {} and creatinine in the blood is {}.".format(sofa[5], collection_dict["Urine"], collection_dict["Creatinine Blood"])
    if True:
        text+= " To summarize: the patient has a total SOFA score of {}.".format(sum(sofa))
    return text
def generate_synthetic_diagnosis_alternate(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf):
    pass


def generate_synthetic_sofa_diff(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf, susinf_collector, septic):

    difflist = []
    index = 0
    if sofa[index] != savesofa[index]:
        difflist.append(("CNS sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 1
    if sofa[index] != savesofa[index]:
        difflist.append(("Cardiovascular sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 2
    if sofa[index] != savesofa[index]:
        difflist.append(("Respiratory sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 3
    if sofa[index] != savesofa[index]:
        difflist.append(("Coagulation sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 4
    if sofa[index] != savesofa[index]:
        difflist.append(("Liver sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 5
    if sofa[index] != savesofa[index]:
        difflist.append(("Renal sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    text = "Given all the information, how will the sofa values change in 24 hours compared to now?"
    if difflist:
        for diffelement in difflist:
            text += " {} will change from {} to {}, this means a change of {}.".format(diffelement[0], diffelement[2], diffelement[3], diffelement[1])
    else:
        text += " No sofa change is expected to occur."
    return text



def generate_synthetic_sepsislabel(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf, susinf_collector, septic):

    collection_dict = {x:round(y, 3) for x,y in collection_dict.items()}
    old_collection_dict = {x:round(y, 3) for x,y in collection_dict.items()}
    text = ""

    if any(susinf_collector[-2:]):
        text+= "Doctors suspect an infection, based on this information and the other information in this text, will the patient be classified as septic tomorrow?"
    else:
        text+= "The doctors don't suspect an infection, based on this information and the other information in this text, will the patient be classified as septic tomorrow?"
    difflist = []
    index = 0
    if sofa[index] > savesofa[index]:
        difflist.append(("CNS sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 1
    if sofa[index] > savesofa[index]:
        difflist.append(("cardiovascular sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 2
    if sofa[index] > savesofa[index]:
        difflist.append(("lung sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 3
    if sofa[index] > savesofa[index]:
        difflist.append(("blood sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 4
    if sofa[index] > savesofa[index]:
        difflist.append(("liver sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 5
    if sofa[index] > savesofa[index]:
        difflist.append(("kidney sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    if (sum(sofa) - sum(savesofa) >= 2) and any(susinf_collector[-2:]):
        text+= " The patient will be septic due to a "
        savelist = []
        for diffelement in difflist:
            savelist.append("{} (sofa increase by {})".format(diffelement[0].split()[0], diffelement[1]))
        text+= " and ".join(savelist) + " failure."
    else:
        text+= " The patient is not expected to develop sepsis."
    return text



def generate_synthetic_reasoning(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf, susinf_collector, septic, used_codes, used_class, used_rrt):
    if used_codes:
        text = "The patient has an existing precondition given by the ICD-9 codes {}.\n".format(", ".join(used_codes))
    else:
        text = ""
    if used_rrt:
        text+="The patient has ongoing RRT therapy.\n"
    collection_dict = {x:round(y, 3) for x,y in collection_dict.items()}
    old_collection_dict = {x:round(y, 3) for x,y in old_collection_dict.items()}
    disallowed = set()
    if any(susinf_collector[-2:]):
        text+= "Doctors suspect an infection, based on this information and the other information in this text, will the patient be classified as septic tomorrow?\n"
    else:
        text+= "The doctors don't suspect an infection, based on this information and the other information in this text, will the patient be classified as septic tomorrow?\n"
    curr_sofa = savesofa
    curr_collection_dict = old_collection_dict
    text+= "First we need to calculate the SOFA scores given the extracted values. The SOFA scores for the current time are the following: \n"
    if True:
        text+= "The minimum value of GCS_eye is {}, GCS_motor is {} and GCS_verbal is {}, this produces the sum {} and means the CNS SOFA is {}.\n".format( curr_collection_dict["minimum GCS_eye"], curr_collection_dict["minimum GCS_motor"], curr_collection_dict["minimum GCS_verbal"], curr_collection_dict["GCS"], curr_sofa[0])
    if True:
        text+= "Because minimum MAP is {}, max Dopamine is {}, max Dobutamine is {}, max Epinephrine is {} and max Norepinephrine is {} with a patient weight of {} kg, the cardiovascular SOFA is {}.\n".format(curr_collection_dict["minimum MAP"], curr_collection_dict["max_dop"], curr_collection_dict["max_dobu"], curr_collection_dict["max_epi"], curr_collection_dict["max_nore"], curr_collection_dict["weight"], curr_sofa[1])
    if True:
        text+= "Given that minimum PO2 is {} and minimum FiO2 is {} the calculated PAO2FIO2 is {}, this means the respiratory SOFA is {}.\n".format(curr_collection_dict["minimum PO2"], curr_collection_dict["minimum FiO2"], curr_collection_dict["pao2fio2"], curr_sofa[2])
    if True:
        text+= "Because the minimum Platelet count is {} the coagulation SOFA is {}.\n".format( curr_collection_dict["minimum Platelet Count"], curr_sofa[3])
    if True:
        text+= "The maximum Bilirubin (Total) is {} leading to a liver SOFA of {}.\n".format(curr_collection_dict["maximum Bilirubin (Total)"], curr_sofa[4])
    if True:
        text+= "Because total Urine output is {} and maximum creatinine in the blood is {} the renal SOFA is {}.\n".format(curr_collection_dict["sum of Urine"], curr_collection_dict["maximum Creatinine Blood"], curr_sofa[5])
    if True:
        text+= "To summarize: the patient has a total SOFA score of {}.\n".format(sum(x for i,x in enumerate(curr_sofa) if i not in disallowed))
    curr_sofa = sofa
    curr_collection_dict = collection_dict
    text+= "Now we need to calculate the SOFA scores with forecasted values. The SOFA scores in the future based on the forecasted values are the following: \n"
    if True:
        text+= "The minimum value of GCS_eye will be {}, GCS_motor will be {} and GCS_verbal will be {}, this produces the sum {} and means the CNS SOFA will be {}.\n".format( curr_collection_dict["minimum GCS_eye"], curr_collection_dict["minimum GCS_motor"], curr_collection_dict["minimum GCS_verbal"], curr_collection_dict["GCS"], curr_sofa[0])
    if True:
        text+= "Because future minimum MAP will be {}, future max Dopamine will be {}, future max Dobutamine will be {}, future max Epinephrine will be {} and future max Norepinephrine will be {} with a patient weight of {} kg, the cardiovascular SOFA will be {}.\n".format(curr_collection_dict["minimum MAP"], curr_collection_dict["max_dop"], curr_collection_dict["max_dobu"], curr_collection_dict["max_epi"], curr_collection_dict["max_nore"], curr_collection_dict["weight"], curr_sofa[1])
    if True:
        text+= "Given that minimum PO2 will be {} and minimum FiO2 will be {} the forecasted PAO2FIO2 will be {}, this means the respiratory SOFA will be {}.\n".format(curr_collection_dict["minimum PO2"], curr_collection_dict["minimum FiO2"], curr_collection_dict["pao2fio2"], curr_sofa[2])
    if True:
        text+= "Because the minimum Platelet count will be {} the coagulation SOFA is going to be {}.\n".format( curr_collection_dict["minimum Platelet Count"], curr_sofa[3])
    if True:
        text+= "The maximum Bilirubin (Total) will be {} leading to a liver SOFA of {}.\n".format(curr_collection_dict["maximum Bilirubin (Total)"], curr_sofa[4])
    if True:
        text+= "Because Urine output will be {} and maximum creatinine in the blood will be {} the renal SOFA will be {}.\n".format(curr_collection_dict["sum of Urine"], curr_collection_dict["maximum Creatinine Blood"], curr_sofa[5])
    if True:
        text+= "To summarize: the patient will have a future total SOFA score of {}.\n".format(sum(x for i,x in enumerate(curr_sofa) if i not in disallowed))

    difflist = []
    index = 0
    if sofa[index] > savesofa[index] and index not in disallowed:
        difflist.append(("CNS system sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 1
    if sofa[index] > savesofa[index] and index not in disallowed:
        difflist.append(("cardiovascular system sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 2
    if sofa[index] > savesofa[index] and index not in disallowed:
        difflist.append(("respiratory system sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 3
    if sofa[index] > savesofa[index] and index not in disallowed:
        difflist.append(("blood system sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 4
    if sofa[index] > savesofa[index] and index not in disallowed:
        difflist.append(("liver sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    index = 5
    if sofa[index] > savesofa[index] and index not in disallowed:
        difflist.append(("kidney sofa", sofa[index] - savesofa[index], savesofa[index], sofa[index]))
    if difflist:
        text+= "This calculation means that the patient will likely experience a "
        savelist = []
        for diffelement in difflist:
            savelist.append("{} failure since SOFA increased by {}".format(" ".join(diffelement[0].split()[:-1]), diffelement[1]))
        text+= ", and ".join(savelist) + "."
    sofa_sum = sum(x for i, x in enumerate(sofa) if i not in disallowed)
    savesofa_sum = sum(x for i, x in enumerate(savesofa) if i not in disallowed)
    if (sofa_sum) - (savesofa_sum) >= 2 and any(susinf_collector[-2:]):
        text+="\nThe patient will develop sepsis in the next 24 hours, because total SOFA increased by {} and infection is suspected".format(sum(x for i,x in enumerate(sofa) if i not in disallowed)-sum(x for i,x in enumerate(savesofa) if i not in disallowed))
    if (sofa_sum) - (savesofa_sum) >= 2 and not any(susinf_collector[-2:]):
        text+="\nThe patient will not develop sepsis in the next 24 hours, because total SOFA increased by {} and infection is not suspected".format(sum(x for i,x in enumerate(sofa) if i not in disallowed)-sum(x for i,x in enumerate(savesofa) if i not in disallowed))
    if (sofa_sum) - (savesofa_sum) < 2 and any(susinf_collector[-2:]):
        text+="\nThe patient will not develop sepsis in the next 24 hours, because total SOFA changed only by {} and infection is suspected".format(sum(x for i,x in enumerate(sofa) if i not in disallowed)-sum(x for i,x in enumerate(savesofa) if i not in disallowed))
    if (sofa_sum) - (savesofa_sum) < 2 and  not any(susinf_collector[-2:]):
        text+="\nThe patient will not develop sepsis in the next 24 hours, because total SOFA changed only by {} and infection is not suspected".format(sum(x for i,x in enumerate(sofa) if i not in disallowed)-sum(x for i,x in enumerate(savesofa) if i not in disallowed))

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

start=timer.time()
new_information = []
#for ts_ind in ts.ts_ind.unique():
#for ts_ind in [0.0]:
import tqdm 
"""
preconditions = ["J40", "J41", "J42", #lung
                 "N17", "N18.9", "N28", #kidney
                 "D68.4", "D68.5", #blood
                  "K70.0", "K70.3", "K70.41", #iver
                  "I50.0", "I50.9", ""]

ex_preconditions = ["J40", "J41", "J42", "J20", "J21",#lung
                 "N17", "N18.9", "N28", "N17.1", "N17.2", #kidney
                 "D68.4", "D68.5", "D68.6", "D68.8", #blood
                  "K70.0", "K70.3", "K70.41", "K70.2", #iver
                  "I50.0", "I50.9", "I50.1", ""]

"""




for ts_ind in tqdm.tqdm(ts.ts_ind.unique(), total=ts.ts_ind.unique().shape[0]):
    #for ts_ind in [0.0]:
    ts_pat = ts[ts["ts_ind"]==ts_ind]
    savesofa = [0, 0, 0, 0, 0, 0]
    old_collection_dict = dict()
    old_susinf = 0
    septic = False
    ts_pat = ts_pat[ts_pat["value"]!=-100.0]
    susinf_collector = [0]

    data_rrt = ts_pat[(ts_pat["variable"] == "rrt")]
    data_icd = ts_pat[(ts_pat["variable"] == "ICD9")]
    icd_list = data_icd.value.unique()
    used_codes, used_clas = categorize_icds(icd_list)
    used_rrt = len(data_rrt) > 0
    for time in [24, 48, 72, 96, 120]:
        *augsofa, augcollection_dict, auglen_measurements = get_aug_sofa(ts_pat, ts_ind, time, old_collection_dict, used_codes, used_clas, used_rrt)
        *sofa, collection_dict, len_measurements = get_sofa(ts_pat, ts_ind, time, used_codes, used_clas, used_rrt)
        if len_measurements == 0:
            break
        susinf = susinfo(ts_pat, ts_ind, time)
        susinf_collector.append(susinf)
        time = time-0.0000001
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "total-sofa", "value": sum(sofa), "TABLE": "analysis", "mean": 0, "std": 1})
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "cns-sofa", "value": sofa[0], "TABLE": "analysis", "mean": 0, "std": 1})
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "cardiovascular-sofa", "value": sofa[1], "TABLE": "analysis", "mean": 0, "std": 1})
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "respiratory-sofa", "value": sofa[2], "TABLE": "analysis", "mean": 0, "std": 1})
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "coagulation-sofa", "value": sofa[3], "TABLE": "analysis", "mean": 0, "std": 1})
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "liver-sofa", "value": sofa[4], "TABLE": "analysis", "mean": 0, "std": 1})
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "renal-sofa", "value": sofa[5], "TABLE": "analysis", "mean": 0, "std": 1})
        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "suspected-infection", "value": susinf, "TABLE": "analysis", "mean": 0, "std": 1})
        """
        texts = generate_synthetic_analysis(ts_pat, ts_ind, time, sofa, susinf)

        for text in texts:
            #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "synthetic_variable_analysis", "TABLE": "TEXT", "textvalue": text})
            #new_information.append({"ts_ind": ts_ind, "hour": time-24, "variable": "synthetic_variable_analysis_timelag", "TABLE": "TEXT", "textvalue": " in 24 hours?".join(text.split("?"))})
            for subtext in text.split("\n"):
                if subtext.strip() != "":
                    new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "synthetic_variable_analysis_partial", "TABLE": "TEXT", "textvalue": subtext})
                    new_information.append({"ts_ind": ts_ind, "hour": time-24, "variable": "synthetic_variable_analysis_partial_timelag", "TABLE": "TEXT", "textvalue": " in 24 hours?".join(subtext.split("?"))})
         
        text  = generate_synthetic_sofa(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf, susinf_collector, septic)

        new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "synthetic_sofa", "TABLE": "TEXT", "textvalue": "What are the sofa scores for the given input and their explanation? " + text})
        new_information.append({"ts_ind": ts_ind, "hour": time-24, "variable": "synthetic_sofa_timelag", "TABLE": "TEXT", "textvalue":"What are the sofa scores in 24 hours given the input, and what are the explanations? " + text})
        
        text = generate_synthetic_sofa_diff(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf, susinf_collector, septic)

        new_information.append({"ts_ind": ts_ind, "hour": time-24, "variable": "synthetic_sofa_diff", "TABLE": "TEXT", "textvalue": text})
        text = generate_synthetic_sepsislabel(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susinf, susinf_collector, septic)
        new_information.append({"ts_ind": ts_ind, "hour": time-24, "variable": "synthetic_sepsis_label", "TABLE": "TEXT", "textvalue": text})
        """
        if susinf and sum(augsofa)-sum(savesofa) >= 2:
            septic=True
        if time > 25:
            text = generate_synthetic_reasoning(savesofa, augsofa, old_collection_dict, augcollection_dict, susinf, old_susinf, susinf_collector, septic, used_codes, used_clas, used_rrt)
            new_information.append({"ts_ind": ts_ind, "hour": time-24, "variable": "synthetic_sepsis_label", "TABLE": "TEXT", "textvalue": text})
        #text = generate_synthetic_diagnosis_alternate(savesofa, sofa, old_collection_dict, collection_dict, susinf, old_susing)
        
        #new_information.append({"ts_ind": ts_ind, "hour": time, "variable": "synthetic_diagnosis_alternate", "TABLE": "TEXT", "TEXTVALUE": text})
        #texts = generate_synthetic_diagnosis(ts_pat, ts_ind, time, sofa, susinf)
        old_collection_dict = collection_dict
        old_susinf = susinf or old_susinf
        savesofa = sofa
new_df = pd.DataFrame(new_information)
ts = pd.concat([ts, new_df]) 
end=timer.time()

ts.sort_values(by=["ts_ind", "hour"], inplace=True)


data = ts[ts["ts_ind"]==0.0]
data.to_csv("for_stefan_patient0_real_icd.csv")
    
pickle.dump([ts, oc, train_ind, valid_ind, test_ind], open('mimic_iii_preprocessed_synthetic_michi_real_icd.pkl', 'wb'))



