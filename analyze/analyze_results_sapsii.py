import sys, re, math, pickle
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Regex patterns must match the gold-text produced by
# generate_synthetic_reasoning_sapsii() in preprocess8_sapsii.py
#
# Gold text template (relevant lines):
#   "The patient's age is {Age_worst}, contributing {N} points."
#   "The worst HR value is {HR_worst}, contributing {N} points."
#   "The worst SBP value is {SBP_worst}, contributing {N} points."
#   "The worst Temperature value is {Temperature_worst}, contributing {N} points."
#   "The worst GCS value is {GCS_worst}, contributing {N} points."
#   "The worst PaO2/FiO2 ratio is {PaO2FiO2_worst}, contributing {N} points."
#   "The worst BUN value is {BUN_worst}, contributing {N} points."
#   "The total Urine output is {Urine_worst}, contributing {N} points."
#   "The worst Sodium value is {Sodium_worst}, contributing {N} points."
#   "The worst Potassium value is {Potassium_worst}, contributing {N} points."
#   "The worst Bicarbonate value is {Bicarbonate_worst}, contributing {N} points."
#   "The worst WBC value is {WBC_worst}, contributing {N} points."
#   "The worst Bilirubin value is {Bilirubin_worst}, contributing {N} points."
#   "the patient has a total SAPS-II score of {total_sapsii}."
#   "The logit is computed as ... = {sapsii_logit}."
#   "The predicted hospital mortality probability is ... = {sapsii_probability}."
# ---------------------------------------------------------------------------

# Named groups for every "worst value" and "contributing points" pair
_PATTERNS = {
    # raw measurements
    "Age":         r"patient's age is (-?\d+\.?\d*)",
    "HR":          r"worst HR value is (-?\d+\.?\d*)",
    "SBP":         r"worst SBP value is (-?\d+\.?\d*)",
    "Temperature": r"worst Temperature value is (-?\d+\.?\d*)",
    "GCS":         r"worst GCS value is (-?\d+\.?\d*)",
    "PaO2FiO2":    r"worst PaO2/FiO2 ratio is (-?\d+\.?\d*)",
    "BUN":         r"worst BUN value is (-?\d+\.?\d*)",
    "Urine":       r"total Urine output is (-?\d+\.?\d*)",
    "Sodium":      r"worst Sodium value is (-?\d+\.?\d*)",
    "Potassium":   r"worst Potassium value is (-?\d+\.?\d*)",
    "Bicarbonate": r"worst Bicarbonate value is (-?\d+\.?\d*)",
    "WBC":         r"worst WBC value is (-?\d+\.?\d*)",
    "Bilirubin":   r"worst Bilirubin value is (-?\d+\.?\d*)",
    # component scores  (match "contributing N points")
    "Age_score":         r"age is -?\d+\.?\d*, contributing (-?\d+) points",
    "HR_score":          r"HR value is -?\d+\.?\d*, contributing (-?\d+) points",
    "SBP_score":         r"SBP value is -?\d+\.?\d*, contributing (-?\d+) points",
    "Temperature_score": r"Temperature value is -?\d+\.?\d*, contributing (-?\d+) points",
    "GCS_score":         r"GCS value is -?\d+\.?\d*, contributing (-?\d+) points",
    "PaO2FiO2_score":    r"PaO2/FiO2 ratio is -?\d+\.?\d*, contributing (-?\d+) points",
    "BUN_score":         r"BUN value is -?\d+\.?\d*, contributing (-?\d+) points",
    "Urine_score":       r"Urine output is -?\d+\.?\d*, contributing (-?\d+) points",
    "Sodium_score":      r"Sodium value is -?\d+\.?\d*, contributing (-?\d+) points",
    "Potassium_score":   r"Potassium value is -?\d+\.?\d*, contributing (-?\d+) points",
    "Bicarbonate_score": r"Bicarbonate value is -?\d+\.?\d*, contributing (-?\d+) points",
    "WBC_score":         r"WBC value is -?\d+\.?\d*, contributing (-?\d+) points",
    "Bilirubin_score":   r"Bilirubin value is -?\d+\.?\d*, contributing (-?\d+) points",
    # totals
    "Total_SAPSII": r"total SAPS-II score of (-?\d+\.?\d*)",
    "Logit":        r"logit is computed as[^=]+=\s*(-?\d+\.?\d*)",
    "Probability":  r"mortality probability is[^=]+=\s*(-?\d+\.?\d*)",
}


def _find(pattern, text, default=-1):
    m = re.findall(pattern, text)
    return m[0] if m else default


def _to_float(val):
    if isinstance(val, str):
        val = val.rstrip(".")
        try:
            return float(val)
        except ValueError:
            return -1.0
    return float(val)


class Output:
    """Parse a SAPS-II reasoning text into a dict of numeric values."""

    def __init__(self, text):
        self.whole_text = text
        self.dict = {}

    def parse_text_into_dict(self):
        text = self.whole_text
        for key, pattern in _PATTERNS.items():
            raw = _find(pattern, text)
            self.dict[key] = _to_float(raw)

    # ------------------------------------------------------------------
    # Level 1 – raw "copy" values (did the LLM read the correct number?)
    # ------------------------------------------------------------------
    COPY_KEYS = ["Age", "HR", "SBP", "Temperature", "GCS",
                 "PaO2FiO2", "BUN", "Urine", "Sodium", "Potassium",
                 "Bicarbonate", "WBC", "Bilirubin"]

    def compare_copy_values(self, other, mod=0.95):
        """Return (matches_list, keys_list) for raw measurement values."""
        matching = []
        for key in self.COPY_KEYS:
            gold_v = self.dict[key]
            pred_v = other.dict[key]
            lo = gold_v * mod - 0.0001
            hi = gold_v * (2 - mod) + 0.0001
            matching.append(1 if lo < pred_v < hi else 0)
        return matching, list(self.COPY_KEYS)

    def compare_copy_values_mse(self, other):
        mse = sum((other.dict[k] - self.dict[k]) ** 2
                  for k in self.COPY_KEYS) / len(self.COPY_KEYS)
        return mse

    # ------------------------------------------------------------------
    # Level 2 – component SAPS-II scores
    # ------------------------------------------------------------------
    SCORE_KEYS = ["Age_score", "HR_score", "SBP_score", "Temperature_score",
                  "GCS_score", "PaO2FiO2_score", "BUN_score", "Urine_score",
                  "Sodium_score", "Potassium_score", "Bicarbonate_score",
                  "WBC_score", "Bilirubin_score"]

    def compare_score_values(self, other, mod=0.95):
        """Return (matches_list, keys_list) for each component point score."""
        matching = []
        for key in self.SCORE_KEYS:
            gold_v = self.dict[key]
            pred_v = other.dict[key]
            lo = gold_v * mod - 0.0001
            hi = gold_v * (2 - mod) + 0.0001
            matching.append(1 if lo < pred_v < hi else 0)
        return matching, list(self.SCORE_KEYS)

    # ------------------------------------------------------------------
    # Level 3 – total SAPS-II, logit, probability
    # ------------------------------------------------------------------
    def compare_total_sapsii(self, other, mod=0.95):
        keys = ["Total_SAPSII", "Logit", "Probability"]
        matching = []
        for key in keys:
            gold_v = self.dict[key]
            pred_v = other.dict[key]
            #if key == "Logit": print(gold_v, pred_v)
            lo = gold_v * mod - 0.0001
            hi = gold_v * (2 - mod) + 0.0001
            #matching.append(1 if lo < pred_v < hi else 0)
            matching.append(1 if abs(pred_v - gold_v) <= abs(gold_v) * (1 - mod) + 0.0001 else 0)
        return matching, keys

    # ------------------------------------------------------------------
    # Internal consistency:
    # Given the raw values the LLM *stated*, do the scores it assigned
    # follow the correct SAPS-II scoring rules?
    # ------------------------------------------------------------------
    @staticmethod
    def _age_score(age):
        if age >= 80: return 18
        if age >= 75: return 16
        if age >= 70: return 15
        if age >= 60: return 12
        if age >= 40: return 7
        return 0

    @staticmethod
    def _hr_score(hr):
        if hr < 40: return 11
        if hr >= 160: return 7
        if 120 <= hr < 160: return 4
        if 40 <= hr < 70: return 2
        return 0

    @staticmethod
    def _sbp_score(sbp):
        if sbp < 70: return 13
        if 70 <= sbp < 100: return 5
        if sbp > 199: return 2
        return 0

    @staticmethod
    def _temp_score(temp):
        return 3 if temp >= 39 else 0

    @staticmethod
    def _gcs_score(gcs):
        if gcs >= 14: return 0
        if gcs >= 11: return 5
        if gcs >= 9: return 7
        if gcs >= 6: return 13
        return 26

    @staticmethod
    def _pf_score(pf):
        if pf < 100: return 11
        if pf < 200: return 9
        return 6

    @staticmethod
    def _bun_score(bun):
        if bun > 83: return 10
        if bun >= 28: return 6
        return 0

    @staticmethod
    def _urine_score(urine):
        if urine < 500: return 11
        if urine < 1000: return 4
        return 0

    @staticmethod
    def _na_score(na):
        if na < 125: return 5
        if na >= 145: return 1
        return 0

    @staticmethod
    def _k_score(k):
        return 5 if (k < 3.0 or k >= 5.0) else 0

    @staticmethod
    def _bicarb_score(b):
        if b < 15: return 6
        if b < 20: return 3
        return 0

    @staticmethod
    def _wbc_score(wbc):
        if wbc < 1.0: return 12
        if wbc >= 20: return 3
        return 0

    @staticmethod
    def _bilir_score(bil):
        if bil >= 6: return 9
        if bil >= 4: return 4
        return 0

    def calculate_inherent_sapsii_calculation(self):
        """
        Re-derive every component score from the raw values the LLM *stated*,
        then check whether those match the scores the LLM also stated.
        Also verify total = sum of stated component scores,
        logit formula, and probability formula.

        Returns (matches_list, keys_list).
        """
        d = self.dict
        recomputed = {
            "Age_score":         self._age_score(d["Age"]),
            "HR_score":          self._hr_score(d["HR"]),
            "SBP_score":         self._sbp_score(d["SBP"]),
            "Temperature_score": self._temp_score(d["Temperature"]),
            "GCS_score":         self._gcs_score(d["GCS"]),
            "PaO2FiO2_score":    self._pf_score(d["PaO2FiO2"]),
            "BUN_score":         self._bun_score(d["BUN"]),
            "Urine_score":       self._urine_score(d["Urine"]),
            "Sodium_score":      self._na_score(d["Sodium"]),
            "Potassium_score":   self._k_score(d["Potassium"]),
            "Bicarbonate_score": self._bicarb_score(d["Bicarbonate"]),
            "WBC_score":         self._wbc_score(d["WBC"]),
            "Bilirubin_score":   self._bilir_score(d["Bilirubin"]),
        }

        matching = []
        keys = list(recomputed.keys())

        for key in keys:
            expected = recomputed[key]
            stated   = d[key]
            matching.append(1 if abs(expected - stated) < 0.5 else 0)

        # Check: stated component scores sum to stated total
        stated_sum = sum(d[k] for k in self.SCORE_KEYS)
        total_ok = 1 if abs(stated_sum - d["Total_SAPSII"]) < 0.5 else 0
        matching.append(total_ok)
        keys.append("Total_SAPSII_sum_check")

        # Check: logit formula applied to stated total
        t = d["Total_SAPSII"]
        if t > -1:  # guard against parse failure
            expected_logit = -7.7631 + 0.0737 * t + 0.9971 * math.log(t + 1)
            logit_ok = 1 if abs(expected_logit - d["Logit"]) < 0.05 else 0
        else:
            logit_ok = 0
        matching.append(logit_ok)
        keys.append("Logit_formula_check")

        # Check: probability = exp(logit)/(1+exp(logit)) matches stated logit
        logit_val = d["Logit"]
        if logit_val > -990 and logit_val < 990:
            #print(logit_val)
            expected_prob = math.exp(logit_val) / (1 + math.exp(logit_val))
            prob_ok = 1 if abs(expected_prob - d["Probability"]) < 0.005 else 0
        else:
            prob_ok = 0
        matching.append(prob_ok)
        keys.append("Probability_formula_check")

        return matching, keys


# ---------------------------------------------------------------------------
# Analysis class – mirrors the SOFA Analysis class
# ---------------------------------------------------------------------------

class Analysis:
    def __init__(self, resultlist):
        self.resultlist = resultlist

    # ---- helpers -----------------------------------------------------------
    def _iter_pairs(self):
        """Yield (gold_Output, pred_Output) skipping parse failures."""
        errors = 0
        for result in self.resultlist:
            gold = Output(result["gold"])
            try:
                pred = Output(result["output"])
            except Exception:
                errors += 1
                continue
            gold.parse_text_into_dict()
            pred.parse_text_into_dict()
            yield gold, pred
        if errors:
            print(f"  [skipped {errors} unparseable outputs]")

    def _iter_pred_only(self):
        """Yield pred_Output only (for internal consistency checks)."""
        errors = 0
        for result in self.resultlist:
            try:
                pred = Output(result["output"])
            except Exception:
                errors += 1
                continue
            pred.parse_text_into_dict()
            yield pred
        if errors:
            print(f"  [skipped {errors} unparseable outputs]")

    # ---- Level 1: raw values -----------------------------------------------
    def analyse_copy_values(self):
        n_keys = len(Output.COPY_KEYS)
        varis = [0] * n_keys
        counter = 0
        for gold, pred in self._iter_pairs():
            out, keys = gold.compare_copy_values(pred)
            counter += 1
            for i in range(n_keys):
                varis[i] += out[i]
        print("=== Raw measurement copy accuracy ===")
        print("variable & accuracy \\\\")
        for i in range(n_keys):
            print(keys[i], "&", round(varis[i] / counter, 3), "\\\\")
        print("-" * 80)

    def analyse_copy_values_mse(self):
        mses = []
        for gold, pred in self._iter_pairs():
            mses.append(gold.compare_copy_values_mse(pred))
        print("=== Raw measurement MSE ===")
        print("Mean MSE:", round(sum(mses) / len(mses), 4))
        print("-" * 80)

    # ---- Level 2: component scores -----------------------------------------
    def analyse_score_values(self):
        n_keys = len(Output.SCORE_KEYS)
        varis = [0] * n_keys
        counter = 0
        for gold, pred in self._iter_pairs():
            out, keys = gold.compare_score_values(pred)
            counter += 1
            for i in range(n_keys):
                varis[i] += out[i]
        print("=== Component score accuracy ===")
        print("variable & accuracy \\\\")
        for i in range(n_keys):
            print(keys[i], "&", round(varis[i] / counter, 3), "\\\\")
        print("-" * 80)

    # ---- Level 3: total SAPS-II, logit, probability ------------------------
    def analyse_total_sapsii(self):
        keys_out = ["Total_SAPSII", "Logit", "Probability"]
        varis = [0] * 3
        counter = 0
        for gold, pred in self._iter_pairs():
            out, keys = gold.compare_total_sapsii(pred)
            counter += 1
            for i in range(3):
                varis[i] += out[i]
        print("=== Total SAPS-II / logit / probability accuracy ===")
        print("variable & accuracy \\\\")
        for i in range(3):
            print(keys_out[i], "&", round(varis[i] / counter, 3), "\\\\")
        print("-" * 80)

    # ---- Internal consistency (no gold needed) -----------------------------
    def analyse_inherent_sapsii_calc(self):
        # Figure out number of keys from first parseable pred
        first_keys = None
        varis = None
        counter = 0
        results = []
        for pred in self._iter_pred_only():
            out, keys = pred.calculate_inherent_sapsii_calculation()
            if first_keys is None:
                first_keys = keys
                varis = [0] * len(keys)
            counter += 1
            for i in range(len(keys)):
                varis[i] += out[i]
            results.append(out)

        print("=== Internal consistency (SAPS-II calculation) ===")
        print("variable & accuracy \\\\")
        for i, key in enumerate(first_keys):
            print(key, "&", round(varis[i] / counter, 3), "\\\\")
        print("-" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_results_sapsii.py <results.pkl>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        resultlist = pickle.load(f)

    analysis = Analysis(resultlist)

    # Level 1 – raw value copying
    analysis.analyse_copy_values()
    analysis.analyse_copy_values_mse()

    # Level 2 – component scores
    analysis.analyse_score_values()

    # Level 3 – total score, logit, probability
    analysis.analyse_total_sapsii()

    # Internal consistency (no gold required)
    analysis.analyse_inherent_sapsii_calc()

