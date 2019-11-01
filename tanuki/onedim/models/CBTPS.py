from tanuki.onedim.models._mixins import *
from tanuki.onedim.models.IBTPS import Inf1DBTPS

#  ██████ ██████  ████████ ██████  ███████ 
# ██      ██   ██    ██    ██   ██ ██      
# ██      ██████     ██    ██████  ███████ 
# ██      ██   ██    ██    ██           ██ 
#  ██████ ██████     ██    ██      ███████ 
# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Cyc1DBTPS(Inf1DBTPS):
    def __init__(self, tensors, bdts, phys_labelss=None):
        self.tensors = CyclicList(tensors)
        self.bdts = CyclicList(bdts)
        if phys_labelss is None:
            phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        self.phys_labelss = CyclicList(phys_labelss)

    def __repr__(self):
        return f"Cyc1DBTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss})"

    def __str__(self, nodata=False):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                bdt = self.bdts[i]
                dataStr += bdt.__str__(nodata=nodata)
                dataStr += "\n"
                tensor = self.tensors[i]
                dataStr += tensor.__str__(nodata=nodata)
                dataStr += ",\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Cyc1DBTPS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        bdts = [self.get_bra_bond(bondsite) for bondsite in range(len(self))]
        return Cyc1DBTPS(tensors, bdts, self.phys_labelss)

    def to_TMS(self):
        return self.to_TPS().to_TMS()

    def to_TPS(self):
        tensors = []
        for i in range(len(self)):
            tensors.append( self.bdts[i][self.get_right_labels_bond(i)] * self.tensors[i][self.get_left_labels_site(i)] )
        from tanuki.onedim.models.CTPS import Cyc1DTPS
        return Cyc1DTPS(tensors, self.phys_labelss)

    def to_BTPS(self):
        return self

    def inner_prod(self, other):
        result = 1.0
        for e in range(len(self)):
            result *= self.get_bra_site(e)
            result *= self.get_bra_bond(e+1)
            result *= other.get_ket_site(e)
            result *= other.get_ket_bond(e+1)
        return result

    def truncate(self, chi, normalize=True, algname="canonize", memo=None, **kwargs):
        if memo is None: memo = {}

        if algname == "naive":
            weight = self.equalize_norms(normalize=normalize)

            for bde in range(len(self)):
                U,S,V = self.bdts[bde].svd(self.get_left_labels_bond(bde), chi=chi, svd_labels=2)
                self.tensors[bde-1] = self.tensors[bde-1] * U
                self.bdts[bde] = S
                self.tensors[bde] = V * self.tensors[bde]

            if normalize:
                for bde in range(len(self)):
                    s = self.bdts[bde].norm()
                    self.bdts[bde] = self.bdts[bde] / s
                    weight *= s

            return weight

        elif algname == "canonize":
            ORIGIN = Cyc1DBTPS(self.tensors, self.bdts, phys_labelss=self.phys_labelss)
            weight = self.universally_canonize(chi=chi, transfer_normalize=normalize)
            ORIGIN_SQ = ORIGIN.inner_prod(ORIGIN).real()
            sqdiff = ( ORIGIN_SQ \
                    - weight * ORIGIN.inner_prod(self).real() * 2 \
                    + weight * weight * self.inner_prod(self).real() ).to_scalar()
            memo["sqdiff"] = sqdiff
            memo["relative_sqdiff"] = sqdiff / ORIGIN_SQ.to_scalar()
            #print(memo)
            return weight

        elif algname == "iterative":
            params = {
                "conv_atol": kwargs.get("conv_atol", 1e-30),
                "conv_rtol": kwargs.get("conv_rtol", 1e-30),
                "max_iter": kwargs.get("max_iter", 2000),
                "initial_value": kwargs.get("initial_value", "canonize_truncation")
                }
            #enough_chi = soujou(self.get_ket_site(0).dims(self.get_phys_labels_site(0)))**(len(self)//2)
            enough_chi = soujou(self.get_ket_site(0).dims(self.get_phys_labels_site(0)))**(len(self)//4)
            if chi is None or chi > enough_chi:
                chi = enough_chi

            weight = self.universally_canonize(chi=chi, transfer_normalize=normalize)

            ORIGIN = self.to_TPS()

            ORIGIN_SQ = 1.0
            for e in range(len(ORIGIN)):
                ORIGIN_SQ *= ORIGIN.get_ket_site(e)
                ORIGIN_SQ *= ORIGIN.get_bra_site(e)
            ORIGIN_SQ = ORIGIN_SQ.real()

            if params["initial_value"] == "naive_truncation":
                PHI = Cyc1DBTPS(self.tensors, self.bdts, self.phys_labelss)
                PHI.truncate(chi=chi, normalize=False, algname="naive")
                PHI = PHI.to_TPS()
            elif params["initial_value"] == "canonize_truncation":
                PHI = Cyc1DBTPS(self.tensors, self.bdts, self.phys_labelss)
                PHI.truncate(chi=chi, normalize=False, algname="canonize")
                PHI = PHI.to_TPS()
            elif params["initial_value"] == "random":
                from tanuki.onedim.models_instant import random_cyc1DTPS
                PHI = random_cyc1DTPS(self.phys_labelss, phys_dimss=[self.tensors[e].dims(self.phys_labelss[e]) for e in range(len(self))], chi=chi)
            #print("PHI",PHI)
            
            sqdiff = float("inf")

            for iteri in range(params["max_iter"]):
                old_sqdiff = sqdiff

                for e in range(len(ORIGIN)):
                    M0 = PHI.get_ket_site(e)

                    A = 1
                    B = 1
                    for i in range(1,len(ORIGIN)):
                        A *= PHI.get_ket_site(e+i)
                        A *= PHI.get_bra_site(e+i)
                        B *= ORIGIN.get_ket_site(e+i)
                        B *= PHI.get_bra_site(e+i)
                    B *= ORIGIN.get_ket_site(e)

                    M = A.solve(B, rows_of_A=PHI.get_bra_left_labels_site(e)+PHI.get_bra_right_labels_site(e), rows_of_B=PHI.get_bra_left_labels_site(e)+PHI.get_bra_right_labels_site(e), assume_a="gen")

                    #print("M0",M0)
                    #print("M",M)

                    M = M * 1.5 - M0 * 0.5
                    PHI.tensors[e] = M

                sqdiff = ( (A * PHI.get_ket_site(e) * PHI.get_bra_site(e)).real() - (B * PHI.get_bra_site(e)).real()*2 + ORIGIN_SQ ).to_scalar()

                if abs(sqdiff-old_sqdiff) <= sqdiff*params["conv_rtol"] + params["conv_atol"]:
                    break

            memo["sqdiff"] = sqdiff
            memo["relative_sqdiff"] = sqdiff / ORIGIN_SQ.to_scalar()
            memo["iter_times"] = iteri+1
            #print(memo)

            PHI = PHI.to_BTPS()
            self.tensors = PHI.tensors
            self.bdts = PHI.bdts
            self.phys_labelss = PHI.phys_labelss

            return weight * self.universally_canonize(chi=None, transfer_normalize=normalize)







