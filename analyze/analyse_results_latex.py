import sys, re, pickle

from sklearn.metrics import confusion_matrix
class Output:
    def __init__(self, text, inputtext=""):
        self.whole_text = text
        self.text1, self.text2  = self.whole_text.split("Now we need to calculate the SOFA scores with forecasted values")
        self.inputtext = inputtext
    def parse_text_into_dict(self):
        self.dict = {}
        self.text = self.text1
        regex, key = "GCS_eye is (\d+\.?\d*)", "GCS_eye"
        """
        print("-"*100)
        print(self.text)
        print(regex)
        print(re.findall(regex, self.text))
        print("-"*100)
        """
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "GCS_motor is (\d+\.?\d*)", "GCS_motor" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "GCS_verbal is (\d+\.?\d*)", "GCS_verbal" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "this produces the sum (\d+\.?\d*)", "GCS_sum" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "means the CNS SOFA is (\d+)", "GCS_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "minimum MAP is (\d+\.?\d*)", "MAP" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Dopamine is (\d+\.?\d*)", "Dopamine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Epinephrine is (\d+\.?\d*)", "Epinephrine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Dobutamine is (\d+\.?\d*)", "Dobutamine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Norepinephrine is (\d+\.?\d*)", "Norepinephrine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "patient weight of (\d+\.?\d*)", "Weight" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "cardiovascular SOFA is (\d+\.?\d*)", "CV_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "PAO2FIO2 is (\d+\.?\d*)", "PAO2FIO2" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "minimum FiO2 is (\d+\.?\d*)", "FIO2" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "minimum PO2 is (\d+\.?\d*)", "PO2" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "respiratory SOFA is (\d+\.?\d*)", "Respiratory_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "Platelet count is (\d+\.?\d*)", "Platelet" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "coagulation SOFA is (\d+\.?\d*)", "Blood_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "Bilirubin \(Total\) is (\d+\.?\d*)", "Bilirubin" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "liver SOFA of (\d+\.?\d*)", "Liver_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "Urine output is (\d+\.?\d*)", "Urine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "creatinine in the blood is (\d+\.?\d*)", "Creatinine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "renal SOFA is (\d+\.?\d*)", "Renal_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "total SOFA score of (\d+\.?\d*)", "Total_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        



        self.text = self.text2
        regex, key = "GCS_eye will be (\d+\.?\d*)", "future GCS_eye" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "GCS_motor will be (\d+\.?\d*)", "future GCS_motor" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "GCS_verbal will be (\d+\.?\d*)", "future GCS_verbal" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "this produces the sum (\d+\.?\d*)", "future GCS_sum" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "means the CNS SOFA will be (\d+)", "future GCS_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "minimum MAP will be (\d+\.?\d*)", "future MAP" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Dopamine will be (\d+\.?\d*)", "future Dopamine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Epinephrine will be (\d+\.?\d*)", "future Epinephrine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Dobutamine will be (\d+\.?\d*)", "future Dobutamine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "max Norepinephrine will be (\d+\.?\d*)", "future Norepinephrine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "patient weight of (\d+\.?\d*)", "future Weight" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "cardiovascular SOFA will be (\d+\.?\d*)", "future CV_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "PAO2FIO2 will be (\d+\.?\d*)", "future PAO2FIO2" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "FiO2 will be (\d+\.?\d*)", "future FIO2" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "PO2 will be (\d+\.?\d*)", "future PO2" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "respiratory SOFA will be (\d+\.?\d*)", "future Respiratory_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "Platelet count will be (\d+\.?\d*)", "future Platelet" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "coagulation SOFA is going to be (\d+\.?\d*)", "future Blood_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "Bilirubin \(Total\) will be (\d+\.?\d*)", "future Bilirubin" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "liver SOFA of (\d+\.?\d*)", "future Liver_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "Urine output will be (\d+\.?\d*)", "future Urine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "creatinine in the blood will be (\d+\.?\d*)", "future Creatinine" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "renal SOFA will be (\d+\.?\d*)", "future Renal_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "future total SOFA score of (\d+\.?\d*)", "future Total_SOFA" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        regex, key = "increased by (-?\d+\.?\d*)", "future increase" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else -1 
        self.dict[key]=result 
        #print(self.dict)
        for key in self.dict:
            #print(key)
            if True:
                if type(self.dict[key]) is str and self.dict[key].endswith("."):
                    self.dict[key]=float(self.dict[key]+"0")
                else:
                    self.dict[key]=float(self.dict[key])
               
        regex, key = "a(nd)? (?:.+?) failure", "Failures"
        result = re.findall(regex, self.text)
        self.dict[key]=result
        
        regex, key = "will(?: not)? develop sepsis", "Sepsis" 
        result = re.findall(regex, self.text)[0] if re.findall(regex, self.text) else "Null"
        if "not" in result:
            self.dict[key]=0
        elif "sepsis" not in result:
            self.dict[key]=-1
        else:
            self.dict[key]=1
        #print(self.dict)
        
    def compare_copy_values(self, other, mod=0.95, prefix=""):
        keys = ["GCS_eye", "GCS_motor", "GCS_verbal", "GCS_sum", "MAP", "Dopamine", "Dobutamine", "Epinephrine", "Norepinephrine", "Weight", "PAO2FIO2", "PO2", "FIO2", "Platelet", "Bilirubin", "Urine", "Creatinine"]

        matching = []
        for key in keys:
            key = prefix+key
            if  other.dict[key] > self.dict[key]*mod-0.0001 and other.dict[key] < self.dict[key]*(2-mod)+0.0001:
                matching.append(1)
            else:
                matching.append(0)
        return matching, [prefix+key for key in keys]

 
    def compare_copy_values_mse(self, other, mod=0.95, prefix=""):
        keys = ["GCS_eye", "GCS_motor", "GCS_verbal", "GCS_sum", "MAP", "Dopamine", "Dobutamine", "Epinephrine", "Norepinephrine", "Weight", "PAO2FIO2", "PO2", "FIO2", "Platelet", "Bilirubin", "Urine", "Creatinine"]

        matching = 0
        for key in keys:
            key = prefix+key
            matching+=(other.dict[key]-self.dict[key])**2
        return matching/len(keys)

    def compare_sofa_values(self, other, mod=0.95, prefix=""):
        keys = ["GCS_SOFA", "CV_SOFA", "Respiratory_SOFA", "Blood_SOFA", "Liver_SOFA", "Renal_SOFA"]

        matching = []
        for key in keys:
            key = prefix+key
            if  other.dict[key] > self.dict[key]*mod-0.0001 and other.dict[key] < self.dict[key]*(2-mod)+0.0001:
                matching.append(1)
            else:
                matching.append(0)
        return matching, [prefix+key for key in keys]

    def compare_sofa_values_mse(self, other, mod=0.95, prefix=""):
        keys = ["GCS_SOFA", "CV_SOFA", "Respiratory_SOFA", "Blood_SOFA", "Liver_SOFA", "Renal_SOFA"]

        matching = 0
        for key in keys:
            key = prefix+key
            matching+=(other.dict[key] - self.dict[key])**2
        return matching/len(keys)

    def compare_total_sofa_values(self, other, mod=0.95, prefix=""):
        keys = ["Total_SOFA", "increase"] if prefix else ["Total_SOFA"]

        matching = []
        for key in keys:
            key = prefix+key
            if  other.dict[key] > self.dict[key]*mod-0.0001 and other.dict[key] < self.dict[key]*(2-mod)+0.0001:
                matching.append(1)
            else:
                matching.append(0)
        return matching, [prefix+key for key in keys]

    def compare_total_sofa_values_mse(self, other, mod=0.95, prefix=""):
        keys = ["Total_SOFA"]

        matching = 0
        for key in keys:
            key = prefix+key
            matching+=(other.dict[key] - self.dict[key])**2
        return matching/len(keys)

    def compare_sepsis_values(self, other, mod=0.95, prefix=""):
        keys = ["Sepsis"]

        true, pred= [], []
        for key in keys:
            key = prefix+key
            true.append(self.dict[key])
            pred.append(other.dict[key])
        return true, pred

    def compare_sofa_systems(self, other):
        #TODOOOOOOO
        set1 = set(self.dict["Failures"])
        set2 = set(other.dict["Failures"])
        try:
            return len(set1&set2)/len(set1)
        except:
            if len(set1)==0 and len(set2)==0: return 1
            else:
                return 0
    def calculate_inherent_sofa_calculation(self, prefix=""):
        gcs_sofa = 0
        if self.dict[prefix+"GCS_sum"] > 14: gcs_sofa = 0
        elif self.dict[prefix+"GCS_sum"] > 12: gcs_sofa = 1 
        elif self.dict[prefix+"GCS_sum"] > 9: gcs_sofa = 2 
        elif self.dict[prefix+"GCS_sum"] > 5: gcs_sofa = 3
        else: gcs_sofa = 4
        
        resp_sofa = 0
        if self.dict[prefix+"PAO2FIO2"] >= 400: resp_sofa = 0
        elif self.dict[prefix+"PAO2FIO2"] >= 300: resp_sofa = 1
        elif self.dict[prefix+"PAO2FIO2"] >= 200: resp_sofa = 2
        elif self.dict[prefix+"PAO2FIO2"] >= 100: resp_sofa = 3
        else: resp_sofa = 4

        coagulation_sofa = 0
        if self.dict[prefix+"Platelet"] > 150: coagulation_sofa = 0
        elif self.dict[prefix+"Platelet"] > 100: coagulation_sofa = 1
        elif self.dict[prefix+"Platelet"] > 50: coagulation_sofa = 2
        elif self.dict[prefix+"Platelet"] > 20: coagulation_sofa = 3
        else: coagulation_sofa = 4

        liver_sofa = 0
        if self.dict[prefix+"Bilirubin"] <= 1.2: liver_sofa = 0
        elif self.dict[prefix+"Bilirubin"] <= 2.0: liver_sofa = 1
        elif self.dict[prefix+"Bilirubin"] <= 6: liver_sofa = 2
        elif self.dict[prefix+"Bilirubin"] <= 12: liver_sofa = 3
        else: liver_sofa = 4

        renal_sofa = 0
        if (self.dict[prefix+"Creatinine"] > 5) or (self.dict[prefix+"Urine"] < 200): renal_sofa = 4
        elif (self.dict[prefix+"Creatinine"] > 3.5) or (self.dict[prefix+"Urine"] < 500): renal_sofa = 3
        elif (self.dict[prefix+"Creatinine"] > 2): renal_sofa = 2
        elif (self.dict[prefix+"Creatinine"] > 1.2): renal_sofa = 1
        else: renal_sofa = 0
        
        cs_sofa = 0
        if self.dict[prefix+"Dopamine"] > 15 or self.dict[prefix+"Epinephrine"] > 0.1 or self.dict[prefix+"Norepinephrine"] > 0.01: cs_sofa = 4
        elif self.dict[prefix+"Dopamine"] > 5 or self.dict[prefix+"Epinephrine"] > 0 or self.dict[prefix+"Norepinephrine"] > 0: cs_sofa = 3
        elif self.dict[prefix+"Dopamine"] > 0 or self.dict[prefix+"Dobutamine"] > 0: cs_sofa = 2
        elif self.dict[prefix+"MAP"] < 70: cs_sofa = 1
        else: cs_sofa = 0

        totalsofa = 0
       
        keys = ["GCS_SOFA", "CV_SOFA", "Respiratory_SOFA", "Blood_SOFA", "Liver_SOFA", "Renal_SOFA"]
        for key in keys:
            totalsofa+=self.dict[prefix+key]
        matching = []
        if gcs_sofa > self.dict[prefix+"GCS_SOFA"] -0.0001 and gcs_sofa < self.dict[prefix+"GCS_SOFA"]+0.0001:
            matching.append(1)
        else:
            matching.append(0)

        if cs_sofa > self.dict[prefix+"CV_SOFA"] -0.0001 and cs_sofa < self.dict[prefix+"CV_SOFA"]+0.0001:
            matching.append(1)
        else:
            matching.append(0)



        if resp_sofa > self.dict[prefix+"Respiratory_SOFA"] -0.0001 and resp_sofa < self.dict[prefix+"Respiratory_SOFA"]+0.0001:
            matching.append(1)
        else:
            matching.append(0)
        
        if coagulation_sofa > self.dict[prefix+"Blood_SOFA"] -0.0001 and coagulation_sofa < self.dict[prefix+"Blood_SOFA"]+0.0001:
            matching.append(1)
        else:
            matching.append(0)

        
        if liver_sofa > self.dict[prefix+"Liver_SOFA"] -0.0001 and liver_sofa < self.dict[prefix+"Liver_SOFA"]+0.0001:
            matching.append(1)
        else:
            matching.append(0)

        if renal_sofa > self.dict[prefix+"Renal_SOFA"] -0.0001 and renal_sofa < self.dict[prefix+"Renal_SOFA"]+0.0001:
            matching.append(1)
        else:
            matching.append(0)


        if totalsofa > self.dict[prefix+"Total_SOFA"] -0.0001 and totalsofa < self.dict[prefix+"Total_SOFA"]+0.0001:
            matching.append(1)
        else:
            matching.append(0)
        keys = ["GCS_SOFA", "CV_SOFA", "Respiratory_SOFA", "Blood_SOFA", "Liver_SOFA", "Renal_SOFA", "Total_SOFA"]
        return matching, [prefix+x for x in keys]
        #TODO Rest here and implement in Analysis
    def calculate_sepsis_calculation(self):
        totalsofa_now = self.dict["Total_SOFA"]
        totalsofa_future = self.dict["future Total_SOFA"]
        diff = totalsofa_future - totalsofa_now
        susinf = 1 if re.findall("octors suspect", self.inputtext) else 0
        if diff > 1 and susinf:
            sepsis = 1
        else: sepsis = 0
        #print(diff, susinf, self.dict["Sepsis"])
        if self.dict["Sepsis"] > 0.5: 
            pass#rint(self.whole_text)
        return sepsis == self.dict["Sepsis"] 
        #TODO implement in Analysis
class Analysis:
    def __init__(self, resultlist):
        self.resultlist = resultlist
    def analyse_copy_values(self, prefix=""):
        varis = [0 for _ in range(17)]
        counter = 0
        errors = 0
        for result in resultlist:
            #print(result["gold"])
            gold_output = Output(result["gold"])
            try:
                pred_output = Output(result["output"])
            except:
                errors+=1
                continue
            gold_output.parse_text_into_dict()
            pred_output.parse_text_into_dict()
            out, keys = gold_output.compare_copy_values(pred_output, prefix=prefix)
            counter+=1
            for i in range(17):
                varis[i]+=out[i]
            #print(result)
        print(prefix+"copy values analysis")
        print("variable & accuracy \\\\")
        for i in range(17):
            print(keys[i], "&", round(varis[i]/counter, 3), "\\\\")
        print("-"*100) 
    

    def analyse_sofa_values(self, prefix=""):
        varis = [0 for _ in range(6)]
        counter = 0
        errors = 0
        for result in resultlist:
            #print(result["gold"])
            gold_output = Output(result["gold"])
            try:
                pred_output = Output(result["output"])
            except:
                errors+=1
                continue
            gold_output.parse_text_into_dict()
            pred_output.parse_text_into_dict()
            out, keys = gold_output.compare_sofa_values(pred_output, prefix=prefix)
            counter+=1
            for i in range(6):
                varis[i]+=out[i]
        print(prefix+"partial sofa values analysis")
        print("variable & accuracy \\\\")
        for i in range(6):
            print(keys[i], "&", round(varis[i]/counter, 3))
        print("-"*100) 


    
    def analyse_total_sofa_values(self, prefix=""):
        varis = [0 for _ in range(2)] if prefix else [0 for _ in range(1)]
        counter = 0
        errors = 0
        var = 2 if prefix else 1
        for result in resultlist:
            #print(result["gold"])
            gold_output = Output(result["gold"])
            try:
                pred_output = Output(result["output"])
            except:
                errors+=1
                continue
            gold_output.parse_text_into_dict()
            pred_output.parse_text_into_dict()
            out, keys = gold_output.compare_total_sofa_values(pred_output, prefix=prefix)
            counter+=1
            for i in range(var):
                varis[i]+=out[i]
        print(prefix+"total sofa values analysis")
        for i in range(var):
            print(keys[i], "&", round(varis[i]/counter, 3), "\\\\")
        print("-"*100)

    def analyse_sofa_systems(self):
        errors = 0
        mses = []
        for result in resultlist:
            #print(result["gold"])
            gold_output = Output(result["gold"])
            try:
                pred_output = Output(result["output"])
            except:
                errors+=1
                continue
            gold_output.parse_text_into_dict()
            pred_output.parse_text_into_dict()
            out = gold_output.compare_sofa_systems(pred_output)
            mses.append(out)
        print("total sofa systems overlap", sum([x for x  in mses if x!=-1])/len([x for x in mses ]))
        print("-"*100)



    def analyse_sepsis_values(self, prefix=""):
        errors = 0
        alltrue, allpred = [], [] 
        from sklearn.metrics import precision_recall_fscore_support
        for result in resultlist:
            #print(result["gold"])
            gold_output = Output(result["gold"], result["input"])
            try:
                pred_output = Output(result["output"], result["input"])
            except:
                errors+=1
                continue
            gold_output.parse_text_into_dict()
            pred_output.parse_text_into_dict()
            true, pred = gold_output.compare_sepsis_values(pred_output)
            alltrue.extend(true)
            allpred.extend(pred)
            #if pred==true: print("BLAAA", pred, true)
        print(confusion_matrix(alltrue, allpred))
        print(precision_recall_fscore_support(alltrue, allpred))
        print("-"*100)

    def analyse_inherent_sofa_calc(self, prefix=""):
        varis = [0 for _ in range(7)]
        counter = 0
        errors = 0
        for result in resultlist:
            #print(result["gold"])
            #gold_output = Output(result["gold"])
            try:
                pred_output = Output(result["output"])
            except:
                errors+=1
                continue
            pred_output.parse_text_into_dict()
            out, keys = pred_output.calculate_inherent_sofa_calculation(prefix=prefix)
            counter+=1
            for i in range(7):
                varis[i]+=out[i]
        print(prefix+"sofa calculation")
        for i in range(7):
            print(keys[i], "&", round(varis[i]/counter, 3), "\\\\")
        print("-"*100) 
    

    def analyse_sepsis_calc(self):
        errors = 0
        mses = []
        for result in resultlist:
            #print(result["gold"])
            #gold_output = Output(result["gold"])
            try:
                pred_output = Output(result["output"], result["input"])
            except:
                errors+=1
                continue
            pred_output.parse_text_into_dict()
            out = pred_output.calculate_sepsis_calculation()
            mses.append(out)
        print("total seps calculation comparison", sum([x for x  in mses if x!=-1])/len([x for x in mses if x!=-1]))
        print("-"*100)





with open(sys.argv[1], "rb") as infile:
    resultlist = pickle.load(infile)

analysis = Analysis(resultlist)
analysis.analyse_copy_values()
analysis.analyse_copy_values(prefix="future ")
analysis.analyse_copy_values_mse()
analysis.analyse_copy_values_mse(prefix="future ")
analysis.analyse_sofa_values()
analysis.analyse_sofa_values(prefix="future ")
#analysis.analyse_sofa_values_mse()
#analysis.analyse_sofa_values_mse(prefix="future ")
analysis.analyse_total_sofa_values()
analysis.analyse_total_sofa_values(prefix="future ")
#analysis.analyse_total_sofa_values_mse()
#analysis.analyse_total_sofa_values_mse(prefix="future ")
analysis.analyse_sofa_systems()
analysis.analyse_sepsis_values()
analysis.analyse_inherent_sofa_calc()
analysis.analyse_inherent_sofa_calc(prefix="future ")
analysis.analyse_sepsis_calc()
