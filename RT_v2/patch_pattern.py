# =======================
# Mitsuba / drjit 
# =======================
import mitsuba as mi
import drjit as dr

# =======================
# Sionna RT
# =======================
from sionna.rt import (
    AntennaPattern,
)

class Parch_Pattern(AntennaPattern):
    def vertical_cut(self, theta: mi.Float) -> mi.Float:
        theta_3dB = dr.deg2rad(125.0)
        SLA_v = 22.5
        return -dr.minimum(12 * dr.square((theta - dr.deg2rad(90)) / theta_3dB), SLA_v)

    def horizontal_cut(self, phi: mi.Float) -> mi.Float:
        phi_3dB = dr.deg2rad(125.0)
        A_max = 22.5
        return -dr.minimum(12 * dr.square(phi / phi_3dB), A_max)

    def combined_pattern(self, theta: mi.Float, phi: mi.Float) -> mi.Float:
        A_max = 22.5
        a_v = self.vertical_cut(theta)
        a_h = self.horizontal_cut(phi)
        total = a_v + a_h
        return -dr.minimum(-total, A_max)

    def __init__(self, polarization: str = "V"):
        if polarization not in {"V", "H", "VH"}:
            raise ValueError("Polarization must be 'V', 'H', or 'VH'")
        self.polarization = polarization

        def my_pattern(theta, phi):
            gain_dB = self.combined_pattern(theta, phi)
            gain_linear = dr.power(10.0, gain_dB / 20.0)

            if self.polarization == "V":
                c_theta = mi.Complex2f(gain_linear, dr.zeros(mi.Float, dr.width(theta)))
                c_phi = mi.Complex2f(dr.zeros(mi.Float, dr.width(phi)), dr.zeros(mi.Float, dr.width(phi)))
            elif self.polarization == "H":
                c_theta = mi.Complex2f(dr.zeros(mi.Float, dr.width(theta)), dr.zeros(mi.Float, dr.width(theta)))
                c_phi = mi.Complex2f(gain_linear, dr.zeros(mi.Float, dr.width(phi)))
            else:  # "VH"
                scale = gain_linear / dr.sqrt(2.0)
                c_theta = mi.Complex2f(scale, dr.zeros(mi.Float, dr.width(theta)))
                c_phi = mi.Complex2f(scale, dr.zeros(mi.Float, dr.width(phi)))

            return c_theta, c_phi

        self.patterns = [lambda theta, phi: my_pattern(theta, phi)]
        