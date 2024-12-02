# src/utils/target_labels.py

class TargetLabels:
    EXP_BAND_GAP = "exp_band_gap_2"
    EXP_FORMATION_ENTHALPY = "exp_formation_enthalpy"
    HSE06_BAND_GAP = "hse06_band_gap"
    PBE_PLUS_U_BAND_GAP = "pbe_+u_band_gap"
    PBE_PLUS_U_FORMATION_ENTHALPY = "pbe_+u_formation_enthalpy"

    @staticmethod
    def get_all_targets():
        return {
            "exp_bg": TargetLabels.EXP_BAND_GAP,
            "exp_fe": TargetLabels.EXP_FORMATION_ENTHALPY,
            "hse06": TargetLabels.HSE06_BAND_GAP,
            "pbe_u": TargetLabels.PBE_PLUS_U_BAND_GAP,
            "pbe_fe": TargetLabels.PBE_PLUS_U_FORMATION_ENTHALPY
        }
