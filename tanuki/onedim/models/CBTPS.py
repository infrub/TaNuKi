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

    def truncate(self, chi, normalize=True, algname="canonize", memo=None, **kwargs):
        if memo is None: memo = {}

        if algname == "naive":
            if normalize:
                weight = 1.0
                for bde in range(len(self)):
                    U,S,V = self.bdts[bde].svd(self.get_left_labels_bond(bde), chi=chi, svd_labels=2)
                    weight *= S.normalize(inplace=True)
                    self.tensors[bde-1] = self.tensors[bde-1] * U
                    self.bdts[bde] = S
                    self.tensors[bde] = V * self.tensors[bde]
                return weight
            else:
                for bde in range(len(self)):
                    U,S,V = self.bdts[bde].svd(self.get_left_labels_bond(bde), chi=chi, svd_labels=2)
                    self.tensors[bde-1] = self.tensors[bde-1] * U
                    self.bdts[bde] = S
                    self.tensors[bde] = V * self.tensors[bde]
                return 1.0

        elif algname == "canonize":
            return self.universally_canonize(chi=chi, transfer_normalize=normalize)

        elif algname == "iterative":
            params = {
                "conv_atol": kwargs.get("conv_atol", 1e-30),
                "conv_rtol": kwargs.get("conv_rtol", 1e-30),
                "max_iter": kwargs.get("max_iter", 600),
                "initial_value": kwargs.get("initial_value", "random")
                }

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
            elif params["initial_value"] == "random":
                from tanuki.onedim.models_instant import random_cyc1DTPS
                PHI = random_cyc1DTPS(self.phys_labelss, phys_dimss=[self.tensors[e].dims(self.phys_labelss[e]) for e in range(len(self))], chi=chi)
            
            sqdiff = float("inf")

            for iteri in range(params["max_iter"]):
                if iteri%10==0: print(iteri, sqdiff)
                old_sqdiff = sqdiff

                for e in range(len(ORIGIN)):
                    M = PHI.get_ket_site(e)
                    Mshape = M.dims(PHI.get_ket_left_labels_site(e)+PHI.get_ket_right_labels_site(e)+PHI.get_phys_labels_site(e))
                    Mlabels = PHI.get_ket_left_labels_site(e)+PHI.get_ket_right_labels_site(e)+PHI.get_phys_labels_site(e)
                    M0 = M

                    B = 1
                    C = 1
                    for i in range(1,len(ORIGIN)):
                        B *= PHI.get_ket_site(e+i)
                        B *= PHI.get_bra_site(e+i)
                        C *= ORIGIN.get_ket_site(e+i)
                        C *= PHI.get_bra_site(e+i)
                    C *= ORIGIN.get_ket_site(e)

                    Bdata = B.to_matrix(PHI.get_bra_left_labels_site(e)+PHI.get_bra_right_labels_site(e), PHI.get_ket_left_labels_site(e)+PHI.get_ket_right_labels_site(e))
                    Cdata = C.to_matrix(PHI.get_bra_left_labels_site(e)+PHI.get_bra_right_labels_site(e), PHI.get_phys_labels_site(e))

                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            Mdata = xp.linalg.solve(Bdata, Cdata, assume_a="pos")
                            M = tnc.matrix_to_tensor(Mdata, Mshape, Mlabels)
                            M = M * 1.5 - M0 * 0.5
                            PHI.tensors[e] = M
                        except:
                            continue

                sqdiff = ( (B * PHI.get_ket_site(e) * PHI.get_bra_site(e)).real() - (C * PHI.get_bra_site(e)).real()*2 + ORIGIN_SQ ).to_scalar()

                if abs(sqdiff-old_sqdiff) <= sqdiff*params["conv_rtol"] + params["conv_atol"]:
                    break

            memo["sqdiff"] = sqdiff
            memo["iter_times"] = iteri+1
            #print(memo)

            PHI = PHI.to_BTPS()
            self.tensors = PHI.tensors
            self.bdts = PHI.bdts
            self.phys_labelss = PHI.phys_labelss

            return self.universally_canonize(chi=None, transfer_normalize=normalize)







