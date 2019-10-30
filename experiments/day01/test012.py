import sys
sys.path.append('../../')
from tanuki import *
import numpy as np



class TestingUnbridgeBondEnv:
    def __init__(self,tensor,ket_left_labels,ket_right_labels):
        self.tensor = tensor
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = aster_labels(ket_left_labels)
        self.bra_right_labels = aster_labels(ket_right_labels)


    def optimal_truncate(self, sigma0, chi=20, maxiter=1000, conv_atol=1e-10, conv_rtol=1e-10, memo=None, funi=0):
        ket_ms_label = unique_label()
        ket_sn_label = unique_label()
        bra_ms_label = aster_label(ket_ms_label)
        bra_sn_label = aster_label(ket_sn_label)

        ETA = self.tensor

        def optimize_M_from_S_N(S,N):
            Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
            Nh = N.adjoint(ket_sn_label, self.ket_right_labels, style="aster")
            B = ETA * Nh * Sh
            C = B * sigma0
            B = S * N * B
            Mshape = B.dims(self.ket_left_labels+[ket_ms_label])
            #assert B.is_hermite(self.ket_left_labels+[ket_ms_label])
            B = B.to_matrix(self.bra_left_labels+[bra_ms_label], self.ket_left_labels+[ket_ms_label])
            C = C.to_vector(self.bra_left_labels+[bra_ms_label])
            M = xp.linalg.solve(B, C, assume_a="pos")

            M = vector_to_tensor(M, Mshape, self.ket_left_labels+[ket_ms_label])
            return M

        def optimize_N_from_M_S(M,S):
            Mh = M.adjoint(self.ket_left_labels, ket_ms_label, style="aster")
            Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
            B = Sh * Mh * ETA
            C = B * sigma0
            B = B * M * S
            Nshape = B.dims([ket_sn_label]+self.ket_right_labels)
            B = B.to_matrix([bra_sn_label]+self.bra_right_labels, [ket_sn_label]+self.ket_right_labels)
            C = C.to_vector([bra_sn_label]+self.bra_right_labels)
            N = xp.linalg.solve(B, C, assume_a="pos")
            N = vector_to_tensor(N, Nshape, [ket_sn_label]+self.ket_right_labels)
            return N

        M,S,N = tensor_svd(sigma0, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = [ket_ms_label, ket_sn_label])

        is_crazy_singular = False
        for iteri in range(maxiter):
            oldS = S
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    M = optimize_M_from_S_N(S,N)
                    N = optimize_N_from_M_S(M,S)
                except (xp.linalg.LinAlgError, xp.linalg.misc.LinAlgWarning) as e:
                    # Let (b,chi)=M.shape, n = rank(ENV).
                    # When b*chi > n, LinAlgError("matrix is singular") or LinAlgWarning("Ill-conditioned matrix: result may not be accurate") occurs, it means "no sufficient terms to decide M,S,N" then "with more small chi I can optimize M,S,N", so deal by shrinking chi.
                    # "no sufficient terms to decide M,S,N" => "with more small chi I can optimize M,S,N" is proven.
                    # Note: converse proposition does NOT work! (so chi can be wasteful even when the program did not storm in this block)
                    # the proof written by infrub is in test0111. need publishing? #TODO)

                    # Therefore finally the result become
                    # ((M*S*N)-sigma0).norm() != 0
                    # (((M*S*N)-sigma0)*H).norm() == 0 (ETA=H*H.adjoint)
                    if chi == 1:
                        # When b > n
                        # Note: It converges immediately (= in this iteri) #TODO why??
                        is_crazy_singular = True
                    else:
                        chi -= 1
                        M.truncate_index(ket_ms_label,chi,inplace=True)
                        S.truncate_index(ket_ms_label,chi,inplace=True)
                        N.truncate_index(ket_sn_label,chi,inplace=True)
                        continue
            if funi==1:
                print()
                print(iteri)
                print(iteri)
                print(iteri)
                print("M", M, M.norm())
                print("S", S)
                print("N", N, N.norm())
            M,S,N = tensor_svd(M*S*N, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = [ket_ms_label, ket_sn_label])
            if funi==1:
                print("M", M, M.norm())
                print("S", S)
                print("N", N, N.norm())
            if funi==2:
                print("M", M, M.norm())
                #print("S", S)
                #print("N", N, N.norm())
            if S.__eq__(oldS, check_atol=conv_atol, check_rtol=conv_rtol):
                break

        if memo is None:
            memo = {}
        memo["iter_times"] = iteri
        memo["is_crazy_singular"] = is_crazy_singular
        memo["chi"] = chi


        return M,S,N




def test0120():
    def f(b,n,chi):
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = TestingUnbridgeBondEnv(V, ["kl"], ["kr"])

        memo = {}
        M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo, funi=1)
        nrm = ((M*S*N-sigma0)*H).norm()

    f(5,5,5)
    # iteri=4, tumari hajimete chi=1 ni nattatoki youyaku seikou.
    # seikouji M.norm()!=1, N.norm()==1 ni natteru. #TODO why?
    # sonotugi iteri=5 ha M,N kawarazu shuusoku hantei. #TODO why?

    # 5,3,1: sugu
    # 5,5,9: sugu
    # 5,5,1: sugu
    # 5,5,9: sugu
    # 5,10,1 : nagai
    # 5,10,2 : sugu
    # 5,10,3 : sugu
    # 5,20,3 : nagai
    # 5,20,4 : sugu



def test0121():
    def f(b,n,chi):
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = TestingUnbridgeBondEnv(V, ["kl"], ["kr"])

        memo = {}
        M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo, funi=2)
        nrm = ((M*S*N-sigma0)*H).norm()

    f(5,5,5)
    # iteri=4, tumari hajimete chi=1 ni nattatoki youyaku seikou.
    # seikouji M.norm()!=1, N.norm()==1 ni natteru. #TODO why?
    # sonotugi iteri=5 ha M,N kawarazu shuusoku hantei. #TODO why?

    # 5,3,1: sugu
    # 5,5,9: sugu
    # 5,5,1: sugu
    # 5,5,9: sugu
    # 5,10,1 : nagai
    # 5,10,2 : sugu
    # 5,10,3 : sugu
    # 5,20,3 : nagai
    # 5,20,4 : sugu



def test0122():
    b = 5
    n = 5
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    sigma0 = random_tensor((b,b),["kl","kr"])
    ETA = V
    ket_left_labels = ["kl"]
    ket_right_labels = ["kr"]
    bra_left_labels = aster_labels(ket_left_labels)
    bra_right_labels = aster_labels(ket_right_labels)

    ket_ms_label = unique_label()
    ket_sn_label = unique_label()
    bra_ms_label = aster_label(ket_ms_label)
    bra_sn_label = aster_label(ket_sn_label)

    def optimize_M_from_S_N(S,N):
        Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
        Nh = N.adjoint(ket_sn_label, ket_right_labels, style="aster")
        B = ETA * Nh * Sh
        C = B * sigma0
        B = S * N * B
        Mshape = B.dims(ket_left_labels+[ket_ms_label])
        #assert B.is_hermite(ket_left_labels+[ket_ms_label])
        B = B.to_matrix(bra_left_labels+[bra_ms_label], ket_left_labels+[ket_ms_label])
        C = C.to_vector(bra_left_labels+[bra_ms_label])
        M = xp.linalg.solve(B, C, assume_a="pos")

        M = vector_to_tensor(M, Mshape, ket_left_labels+[ket_ms_label])
        return M

    def optimize_N_from_M_S(M,S):
        Mh = M.adjoint(ket_left_labels, ket_ms_label, style="aster")
        Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
        B = Sh * Mh * ETA
        C = B * sigma0
        B = B * M * S
        Nshape = B.dims([ket_sn_label]+ket_right_labels)
        B = B.to_matrix([bra_sn_label]+bra_right_labels, [ket_sn_label]+ket_right_labels)
        C = C.to_vector([bra_sn_label]+bra_right_labels)
        N = xp.linalg.solve(B, C, assume_a="pos")
        N = vector_to_tensor(N, Nshape, [ket_sn_label]+ket_right_labels)
        return N

    def printer(M,S,N,text):
        print()
        print(text)
        print()
        print(M.data)
        print(M.norm())
        print()
        print(S.data)
        print(S.norm())
        print()
        print(N.data.transpose())
        print(N.norm())
        print()

    M,S,N = tensor_svd(random_tensor_like(sigma0),ket_left_labels, chi=1, svd_labels=[ket_ms_label,ket_sn_label])
    printer(M,S,N,"init")
    M = optimize_M_from_S_N(S,N)
    printer(M,S,N,"optimize M")
    N = optimize_N_from_M_S(M,S)
    printer(M,S,N,"optimize N")
    M,S,N = tensor_svd(M*S*N,ket_left_labels, chi=1, svd_labels=[ket_ms_label,ket_sn_label])
    printer(M,S,N,"SVD")
    M = optimize_M_from_S_N(S,N)
    printer(M,S,N,"optimize M")
    N = optimize_N_from_M_S(M,S)
    printer(M,S,N,"optimize N")
    M,S,N = tensor_svd(M*S*N,ket_left_labels, chi=1, svd_labels=[ket_ms_label,ket_sn_label])
    printer(M,S,N,"SVD",)

    #shokiti sigma0 demo random_tensor demo M3==M4==M5 !!


test0122()
