from tanuki.onedim.models._mixins import *

#  ██████  ██████  ████████ ██████  ███████ 
# ██    ██ ██   ██    ██    ██   ██ ██      
# ██    ██ ██████     ██    ██████  ███████ 
# ██    ██ ██   ██    ██    ██           ██ 
#  ██████  ██████     ██    ██      ███████ 
# bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- tensors[len-1] -- bdts[len]
class Obc1DBTPS(Mixin_1DSim_PS, Mixin_1DSimBTPS, MixinObc1DBTP_):
    def __init__(self, tensors, bdts, phys_labelss=None):
        self.tensors = list(tensors)
        self.bdts = list(bdts)
        if len(self.bdts)+1==len(self.tensors):
            self.bdts = [tni.dummy_diagonalTensor()] + self.bdts + [tni.dummy_diagonalTensor()]
        if phys_labelss is None:
            self.phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        else:
            self.phys_labelss = list(phys_labelss)

    def __repr__(self):
        return f"Obc1DBTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss})"

    def __str__(self):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                bdt = self.bdts[i]
                dataStr += str(bdt)
                dataStr += "\n"
                tensor = self.tensors[i]
                dataStr += str(tensor)
                dataStr += ",\n"
            bdt = self.bdts[len(self)]
            dataStr += str(bdt)
            dataStr += "\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Obc1DBTPS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        bdts = [self.get_bra_bond(bondsite) for bondsite in range(len(self)+1)]
        return Obc1DBTPS(tensors, bdts, self.phys_labelss)

    def to_tensor(self):
        return self.to_TPS().to_TMS().to_tensor()

    def to_TMS(self):
        return self.to_TPS().to_TMS()

    def to_TPS(self):
        tensors = []
        for i in range(len(self)):
            tensors.append( self.bdts[i][self.get_right_labels_bond(i)] * self.tensors[i][self.get_left_labels_site(i)] )
        tensors[-1] = tensors[-1][self.get_right_labels_site(len(self)-1)] * self.bdts[len(self)][self.get_left_labels_bond(len(self))]
        from tanuki.onedim.models.OTPS import Obc1DTPS
        return Obc1DTPS(tensors, self.phys_labelss)

    def to_BTPS(self):
        return self



    # canonizing methods
    
    # [Fin] bde = 1,2,..,len(self)-1
    # [bde=2]:
    # /-(1)-[1]-(2)-      /-(2)-
    # |      |        ==  |
    # \-(1)-[1]-(2)-      \-(2)-
    def locally_left_canonize_around_not_end_bond(self, bde, chi=None, rtol=None, atol=None):
        U, S, V = tnd.truncated_svd(self.bdts[bde-1]*self.tensors[bde-1]*self.bdts[bde], self.get_left_labels_bond(bde-1)+self.get_phys_labels_site(bde-1), chi=chi, rtol=rtol, atol=atol)
        self.tensors[bde-1] = U/self.bdts[bde-1]
        self.bdts[bde] = S
        self.tensors[bde] = V*self.tensors[bde]
        return 1.0

    # [Fin] bde = 1,..,len(self)-1
    # [bde=3]:
    # -(3)-[3]-(4)-\      -(3)-\
    #       |      |  ==       |
    # -(3)-[3]-(4)-/      -(3)-/
    def locally_right_canonize_around_not_end_bond(self, bde, chi=None, rtol=None, atol=None):
        U, S, V = tnd.truncated_svd(self.bdts[bde]*self.tensors[bde]*self.bdts[bde+1], self.get_left_labels_bond(bde), chi=chi, rtol=rtol, atol=atol)
        self.tensors[bde-1] = self.tensors[bde-1]*U
        self.bdts[bde] = S
        self.tensors[bde] = V/self.bdts[bde+1]
        return 1.0


    def locally_left_canonize_around_right_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        bde = len(self)
        if self.bdts[bde].size==1:
            if end_dealing=="no":
                return 1.0
            elif end_dealing=="normalize":
                w = (self.bdts[bde-1]*self.tensors[bde-1]).norm()
                self.tensors[bde-1] /= w
                return w
            else:
                raise UndecidedError
        else:
            if end_dealing=="no":
                return 1.0
            else:
                raise UndecidedError

    def locally_right_canonize_around_left_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        bde = 0
        if self.bdts[bde].size==1:
            if end_dealing=="no":
                return 1.0
            elif end_dealing=="normalize":
                w = (self.tensors[bde]*self.bdts[bde+1]).norm()
                self.tensors[bde] /= w
                return w
            else:
                raise UndecidedError
        else:
            if end_dealing=="no":
                return 1.0
            else:
                raise UndecidedError


    def locally_left_canonize_around_bond(self, bde, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if bde==0:
            return 1.0
        elif bde==len(self):
            return self.locally_left_canonize_around_right_end(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        else:
            return self.locally_left_canonize_around_not_end_bond(bde, chi=chi, rtol=rtol, atol=atol)

    def locally_right_canonize_around_bond(self, bde, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if bde==0:
            self.locally_right_canonize_around_left_end(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        elif bde==len(self):
            return 1.0
        else:
            return self.locally_right_canonize_around_not_end_bond(bde, chi=chi, rtol=rtol, atol=atol)


    # [bde=2, already=0]
    # /-(0)-[0]-      /-(1)-[1]-      /-
    # |      |    ==  |      |    ==  |
    # \-(0)-[0]-      \-(1)-[1]-      \-
    def globally_left_canonize_upto(self, upto_bde=None, already_bde=0, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if upto_bde is None: upto_bde = len(self)
        weight = 1.0
        for bde in range(already_bde+1, upto_bde+1):
            weight *= self.locally_left_canonize_around_bond(bde, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        return weight

    # [bde=4, already=6]
    # -[4]-(5)-\      -[5]-(6)-\      -\
    #   |      |  ==    |      |  ==   |
    # -[4]-(5)-/      -[5]-(6)-/      -/
    def globally_right_canonize_upto(self, upto_bde=0, already_bde=None, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if already_bde is None: already_bde = len(self)
        weight = 1.0
        for bde in range(already_bde-1, upto_bde-1, -1):
            weight *= self.locally_right_canonize_around_bond(bde, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        return weight

    def universally_canonize(self, left_already=0, right_already=None, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if right_already is None: right_already = len(self)
        weight = 1.0
        weight *= self.globally_left_canonize_upto(right_already, left_already, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        weight *= self.globally_right_canonize_upto(left_already, right_already, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        return weight

    canonize = universally_canonize



    def is_locally_left_canonical_around_bond(self, bde, rtol=1e-5, atol=1e-8):
        if bde == 0:
            return CollateralBool(True, {"factor":1.0})
        S = self.bdts[bde-1]
        V = self.tensors[bde-1]
        SV = S*V
        re = SV.is_prop_semi_unitary(self.get_left_labels_bond(bde-1)+self.get_phys_labels_site(bde-1), rtol=rtol, atol=atol)
        return re

    def is_locally_right_canonical_around_bond(self, bde, rtol=1e-5, atol=1e-8):
        if bde == len(self):
            return CollateralBool(True, {"factor":1.0})
        U = self.tensors[bde]
        S = self.bdts[bde+1]
        US = U*S
        re = US.is_prop_semi_unitary(self.get_phys_labels_site(bde)+self.get_right_labels_bond(bde+1), rtol=rtol, atol=atol)
        return re

    def is_grobally_left_canonical_upto_bond(self, upto_bde=None, rtol=1e-5, atol=1e-8):
        if upto_bde is None: upto_bde = len(self)
        ok = True
        res = {}
        for bde in range(0, upto_bde+1):
            re = self.is_locally_left_canonical_around_bond(bde, rtol=rtol, atol=atol)
            res[bde] = re
            if not re: ok = False
        return CollateralBool(ok, res)

    def is_grobally_right_canonical_upto_bond(self, upto_bde=0, rtol=1e-5, atol=1e-8):
        ok = True
        res = {}
        for bde in range(len(self), upto_bde-1, -1):
            re = self.is_locally_right_canonical_around_bond(bde, rtol=rtol, atol=atol)
            res[bde] = re
            if not re: ok = False
        return CollateralBool(ok, res)

    def is_globally_both_canonical_upto_bond(self, upto_bde, rtol=1e-5, atol=1e-8):
        return self.is_grobally_left_canonical_upto_bond(upto_bde, rtol=rtol, atol=atol) & self.is_grobally_right_canonical_upto_bond(upto_bde, rtol=rtol, atol=atol)

    def is_universally_canonical(self, bde_start=None, bde_end=None, rtol=1e-5, atol=1e-8):
        return self.is_grobally_left_canonical_upto_bond(rtol=rtol, atol=atol) & self.is_grobally_right_canonical_upto_bond(rtol=rtol, atol=atol)

    is_canonical = is_universally_canonical




    # applying methods
    def apply(self, gate, offset=0, chi=None, keep_universal_canonicality=True, keep_phys_labels=True):
        if type(gate) in [Obc1DBTPO, Obc1DTPO, Obc1DTMO]:
            gate = gate.to_BTPO()
        else: # list of gates
            for reallygate in gate:
                self.apply(reallygate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)
            return
        tnop.apply_fin1DSimBTPS_fin1DSimBTPO(self,gate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)

    def apply_everyplace(self, gate, chi=None, keep_universal_canonicality=True, gating_order="grissand"):
        if type(gate) in [Obc1DBTPO, Obc1DTPO, Obc1DTMO]:
            gate = gate.to_BTPO()
        else:
            for reallygate in gate:
                self.apply_everyplace(reallygate,chi=chi,keep_universal_canonicality=keep_universal_canonicality,gating_order=gating_order)
            return
        if gating_order == "grissand":
            for k in range(len(self)-len(gate)+1):
                self.apply(gate, offset=k, chi=chi, keep_universal_canonicality=keep_universal_canonicality, keep_phys_labels=True)
        elif gating_order == "trill":
            for i in range(len(gate)):
                for k in range(i,len(self)-len(gate)+1,len(gate)):
                    self.apply(gate, offset=k, chi=chi, keep_universal_canonicality=keep_universal_canonicality, keep_phys_labels=True)
