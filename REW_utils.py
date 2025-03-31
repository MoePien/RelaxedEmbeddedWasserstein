import numpy as np
import networkx as nx
import ot
import matplotlib.pyplot as plt
import pyvista as pv
import torch
import anytree
from copy import deepcopy
from tqdm import tqdm,trange
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class GM:
    '''    
    Creates a Gauged (Metric) measure space from Euclidean or Surface data.
    '''
    def __init__(self,mode,gauge_mode = None,X = None,g = None,xi = None,Tris=None,Nodes=None,
                 Edges=None,normalize_gauge=False,squared=False):
        if mode == "gauge_only":
            assert type(g) is np.ndarray
            
        elif mode == "euclidean":
            assert gauge_mode == "euclidean" or gauge_mode == "sqeuclidean"
            assert X is not None
                   
        elif mode == "graph" or mode == "weighted_graph":
            assert Edges is not None
        
        elif mode == "surface":
            assert Tris is not None
            assert X is not None
        
        self.X = X
        self.mode = mode
        self.gauge_mode = gauge_mode
        self.normalize_gauge = normalize_gauge
        self.Tris = Tris
        self.Edges = Edges
        self.Nodes = Nodes
        
        #Create Graph if necessary
        if self.mode == "graph" or self.mode == "weighted_graph" or self.mode == "surface":
            if self.mode == "graph":
                self.G = nx.Graph()
                if self.Nodes is not None:
                    self.G.add_nodes_from(self.Nodes)
                self.G.add_edges_from(self.Edges)
            elif self.mode == "weighted_graph":
                self.G = nx.Graph()
                if self.Nodes is not None:
                    self.G.add_nodes_from(self.Nodes)
                self.G.add_weighted_edges_from(self.Edges)
            elif self.mode == "surface":
                self.G = gen_graph_from_surface(self.X,Tris)
            
        self.g = self.set_g(g)
        if squared == True:
            self.g = self.g**2
        self.len = len(self.g)
        self.xi = self.set_xi(xi)
        
    def set_X(self,X):
        return X

    def set_g(self,g):
        if self.mode == "gauge_only":#type(g) == np.ndarray:
            g = g
        elif self.mode == "euclidean":
            g = ot.dist(self.X,metric=self.gauge_mode)
        elif self.mode == "surface" or self.mode == "graph" or self.mode == "weighted_graph":
            if self.gauge_mode == "adjacency":
                g = np.array(nx.adjacency_matrix(self.G).todense(),dtype=float)
            elif self.gauge_mode == "djikstra":
                g = self.djikstra_gauge()
        if self.normalize_gauge:
            g /= np.max(g)
        return g
    
    def set_xi(self,xi):
        if xi is None:
            return ot.unif(self.len)
        elif type(xi) == str and xi == "surface_uniform":
            return surface_uniform_xi(self.X,self.Tris)
        else:
            return xi
        
    def djikstra_gauge(self):
        if self.mode == "weighted_graph" or self.mode == "surface":
            dic = dict(nx.weighted.all_pairs_dijkstra_path_length(self.G))
        elif self.mode == "graph":
            dic = dict(nx.all_pairs_dijkstra_path_length(self.G))
        g = np.zeros((len(self.G.nodes),len(self.G.nodes)))
        for key in dic.keys():
            g[int(key),np.array(list(dic[key].keys()),dtype=int)] = np.array(list(dic[key].values()))
        g = (1/2) * (g + g.T)
        return g
    
def REW(X1,X2,Z,n_its,lambda_GW,eps,MZ=None, use_tqdm=True):
    '''
    Computes the lambda-embedded (or Relaxed Embedded) Wasserstein distance.
    The algorithm is based on reformulation as a regularized unbalanced multi-marginal GW transport problem,
    see also <https://github.com/Gorgotha/UMGW>.
    '''
    kappa = lambda_GW # Alignment with UMGW notation
    if isinstance(kappa, list):
        kappa_ls = kappa
    else:
        kappa_ls = [kappa]
    if MZ is None:
        MZ = ot.dist(Z,metric="euclidean")
    
    zeta = ot.unif(MZ.shape[0])
    Z1,Z2,zeta1,zeta2 = Z,Z,deepcopy(zeta),deepcopy(zeta)

    gamma_X1Y1 = X1.xi[:,None] * zeta1[None,:]
    gamma_Y2X2 = zeta2[:,None] * X2.xi[None,:]

    for kappa in kappa_ls:
        if use_tqdm:
            iterbar = tqdm(range(n_its))
        else:
            iterbar = range(n_its)
        for i in iterbar:
            cGWX1Y = compute_cGW(X1.g,MZ,X1.xi,zeta1,gamma_X1Y1)
            cGWX1Y -= np.min(cGWX1Y)
            cGWX1Y *= kappa
    
            cGWYX2 = compute_cGW(MZ,X2.g,zeta2,X2.xi,gamma_Y2X2)
            cGWYX2 -= np.min(cGWYX2)
            cGWYX2 *= kappa
    
            r = anytree.AnyNode(id=0)
            n1 = anytree.AnyNode(id=1,parent=r)
            n2 = anytree.AnyNode(id=2,parent=n1)
            n3 = anytree.AnyNode(id=2,parent=n2)
    
            forward = [node for node in anytree.PreOrderIter(r)]
    
            #X1
            r.mu = X1.xi
            r.rho = np.inf
            r.cost = None
    
            #Y1
            n1.rho = 0
            n1.cost = torch.from_numpy(cGWX1Y)
    
            #Y2
            n2.rho = 0
            n2.cost = torch.from_numpy(MZ)**2
            n2.cost /= torch.max(n2.cost)
    
            #X2
            n3.rho = np.inf
            n3.mu = X2.xi
            n3.cost = torch.from_numpy(cGWYX2)
            m = max(torch.max(n1.cost),torch.max(n2.cost),torch.max(n3.cost))
            n1.cost /= m
            n2.cost /= m
            n3.cost /= m
    
            r = tree_sinkhorn_torch(r,eps,torch.Tensor([torch.inf]))
    
            zeta1 = r.children[0].marginal
            zeta2 = r.children[0].children[0].marginal
            gamma_X1Y1 = n1.pi_left[:, None] * n1.K * n1.pi_right[None, :]
            gamma_Y2X2 = n3.pi_left[:, None] * n3.K * n3.pi_right[None, :]
    return gamma_X1Y1,gamma_Y2X2,zeta1,zeta2
    
def tree_sinkhorn_torch(root, eps, rho,
    divergence='KL',
    max_iter=10000,
    cvgce_thres=1e-5, 
    n_its_check_cvgce=10,
    pot_large=1e10,
    verbose=False):
    '''
    Source: <https://github.com/jvlindheim>
    
    Unbalanced Sinkhorn algorithm, which computes the multimarginal transport between
    histograms with tree-structured costs. The data is given in form
    of a root node with certain properties (see below).
    As divergences, use Kullback-Leibler.

    See Haasler, Isabel, et al. "Multi-marginal Optimal Transport and Schr\" odinger Bridges on Trees." arXiv preprint arXiv:2004.06909 (2020).

    root: root node of the tree. Every node needs to have certain properties, e.g.:
        cost: Backward directed OT costs, i.e. cost has shape (parent_shape, child_shape), then
        multiplication with the corresponding K (e.g. Kv) leads from the child
        to the parent, so that u_iK_ijv_j = pi_ij, i.e. number of rows matches parent.
        (possibly) mu: histogram (masked array). If not given, barycenter is computed for this node.
        However, root node needs to have the mu parameter and will dictate the shape of the domain,
        which all measures need to have. (This restriction is really only for being able to 
        use the Lebesgue measure as reference measure for the unknowns and readily know its shape.)
    eps: sinkhorn regularization parameter
    rho: unbalanced regularization parameter. 
        If np.inf is given, compute balanced regularized transport. 
        If None is given, individual rhos are used for every measure 
        (useful e.g. for computing barycenters with custom weights), which are saved in the nodes.

    returns:
    root node of tree where now each marginal and marginal plan is computed.
    '''

    assert divergence in ['KL', 'TV'], "unknown divergence"

    # init tree
    assert hasattr(root, 'mu'), "root node needs to be a node with given measure mu or domain_shape must be given"
    forward = [node for node in anytree.PreOrderIter(root)]
    backward = forward[::-1]
    for node in forward:
        assert hasattr(node, 'cost')
        node.given = (hasattr(node, 'mu'))
        if node.cost is not None:
            node.domain_shape = len(node.cost.T)
        else:
            node.domain_shape = len(node.mu)
        node.u = torch.ones(node.domain_shape)
        node.K = [] if node.is_root else torch.exp(-node.cost/eps)
        node.a_forw = torch.ones(node.domain_shape)
        if not node.is_root:
            node.a_backw = torch.ones(node.parent.domain_shape)
        node.marginal = None
        node.pi_left = None
        node.pi_right = None
        if rho is not None:
            node.rho = torch.Tensor([rho])
        node.exponent = node.rho/(node.rho+eps) if not torch.isinf(node.rho) else 1.0

    # init iterations
    update = np.inf
    it = 0
    memsnapit = 10
    prevu = None

    # unbalanced multi-sinkhorn iterations
    while (update > cvgce_thres and it < max_iter):
        del prevu
        prevu = deepcopy([node.u for node in forward])

        # forward pass: update scaling variables u and then update alphas in
        # direction from root down towards leaves simultaneously
        for node in forward:
            incoming = torch.stack([child.a_backw for child in node.children] + ([] if node.is_root else [node.a_forw]))
            if node.given:
                if node.exponent == 0:
                    node.u = torch.ones(torch.shape(node.u))
                    
                elif divergence == 'KL':
                    node.u = ((node.mu / torch.prod(incoming, axis=0))**node.exponent)#*node.itermult
                elif divergence == 'TV':
                    node.u = torch.minimum(torch.exp((rho+0)/eps), torch.maximum(np.exp(-(rho+0)/eps), node.mu/torch.prod(incoming, dim=0)))
            for i, child in enumerate(node.children):
                tmp3 = list(incoming[:i]) + list(incoming[i+1:])
                if len(tmp3) == 0:
                    a_prod = 1
                else:
                    a_prod = torch.prod(torch.stack(tmp3), dim=0)
                child.a_forw = torch.matmul(child.K.T,(node.u*a_prod))

        # backward pass: update alphas in direction from leaves up towards root
        for node in backward[:-1]: # everyone except root
            tmp = [child.a_backw for child in node.children]
            if len(tmp) == 0:
                a_prod = 1
            else:
                a_prod = torch.prod(torch.stack(tmp), dim=0)
            node.a_backw = torch.matmul(node.K,(node.u*a_prod))
            
        it += 1

        # compute updates every couple iterations
        if it % n_its_check_cvgce == 0:
            update = max([torch.abs(node.u - pru).max() \
                          / max(1., (node.u).max(), (pru).max()) \
                        for (node, pru) in zip(forward, prevu)])

            if verbose >= 2:
                print("-----it {0}, update {1}".format(it, update))
            if np.isinf(update):
                print("Algorithm diverged. Return None.")

    # compute marginals and marginal plans
    for node in forward:
        incoming = torch.stack([child.a_backw for child in node.children] + ([] if node.is_root else [node.a_forw]))
        node.marginal = node.u * torch.prod(incoming, dim=0)
        if not node.is_root:
            parent_in = ([] if node.parent.is_root else [node.parent.a_forw]) \
                        + [child.a_backw for child in node.parent.children if child != node] 
            # first line: from parent's parent, second line: from parents children except node
            if len(parent_in) == 0:
                tmp_mult = 1
            else:
                tmp_mult = torch.prod(torch.stack(parent_in), dim=0)
            tmp2 = [child.a_backw for child in node.children]
            if len(tmp2) == 0:
                tmp_mult2 = 1
            else:
                tmp_mult2 = torch.prod(torch.stack(tmp2), dim=0)
            node.pi_left = node.parent.u*tmp_mult
            node.pi_right = node.u*tmp_mult2
    return root

#FUNCTIONS

def area_of_tri(tri):
    p1,p2,p3 = tri
    a = np.linalg.norm(p1-p2)
    b = np.linalg.norm(p2-p3)
    c = np.linalg.norm(p3-p1)
    s = (a+b+c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def surface_uniform_xi(X,Tris):
    mu = np.zeros(len(X))
    for tri in Tris:
        mu[tri] += 1/3 * area_of_tri(X[tri])
    mu /= np.sum(mu)
    return mu

def gen_graph_from_surface(X,Tris):
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    for i in range(len(Tris)):
        tri = Tris[i]
        for l1 in range(3):
            for l2 in range(l1+1,3):
                G.add_edge(tri[l1],tri[l2],weight = np.linalg.norm(X[tri[l1]] - X[tri[l2]]))
    return G
    

def img2atomic(img, reduce=True):
    '''
    Creates a discrete measure from an image.
    '''
    assert img.ndim == 2, "img needs to be 2d array"
    x, y = img.shape
    pts = np.stack([grid.flatten() for grid in np.meshgrid(np.arange(x), y-np.arange(y))], axis=1)
    if reduce:
        supp = np.array(pts[img.flatten() > 0],dtype=float)
        heights = np.array(img.flatten()[img.flatten() > 0],dtype=float)
    else:
        supp = np.array(pts,dtype=float)
        heights = np.array(img.flatten(),dtype=float)
    heights /= np.sum(heights)
    
    return supp,heights


def compute_cGW(MY,MZ,ups,z,P):
    A = np.einsum("ij,j->i", MY ** 2, ups)[:, None]
    B = np.einsum("ij,j->i", MZ ** 2, z)[None,:]
    C = - 2 * np.einsum("ij,kj->ik", MY, np.einsum("kl,jl->kj", MZ, P))
    return A+B+C

def adapt_to_thres(zeta, thres):
    zeta = deepcopy(zeta)
    zeta[zeta < thres] = 0.
    return zeta/zeta.sum()

def manifold_to_GM(manifold):
    ZZ = GM(X=np.array(manifold.points,dtype="d"),Tris=manifold.faces.reshape((-1, 4))[:,1:],
        mode="surface",gauge_mode="djikstra",xi="surface_uniform",normalize_gauge=True,squared=False)
    return ZZ, ZZ.X, ZZ.g



def pairwise_spherical_distances(points):
    """
    Calculate pairwise spherical distances between points on a sphere.

    Parameters:
        points (np.ndarray): A matrix of shape (n, 3) where each row is a point in Euclidean coordinates.

    Returns:
        np.ndarray: A matrix of shape (n, n) with pairwise spherical distances.
    """
    # Normalize points to ensure they lie on the unit sphere
    normalized_points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Compute the cosine of the spherical distances
    cos_distances = np.clip(np.dot(normalized_points, normalized_points.T), -1.0, 1.0)
    
    # Compute the spherical distances using the arccosine
    spherical_distances = np.arccos(cos_distances)
    
    return spherical_distances.astype(np.float64)

def pairwise_euclidean_distances(points):
    """
    Calculate pairwise distances between points/vectors.

    Parameters:
        points (np.ndarray): A matrix of shape (n, 3) where each row is a point in Euclidean coordinates.

    Returns:
        np.ndarray: A matrix of shape (n, n) with pairwise euclidean distances.
    """
    distances = ot.dist(points,metric="euclidean")
    
    return distances.astype(np.float64)

def create_circular_space(n):
    # Due to pyvista implementation, +1 for consistent point numbers
    # almost uniform grid: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    full_sphere = pv.Circle(resolution=(n)**2)
    ZZ, Z, MZ = manifold_to_GM(full_sphere)
    MZ = MZ/MZ.max()
    ZZ.g = MZ.astype(np.float32)
    return full_sphere, ZZ, Z, MZ


def create_spherical_space(n):
    # Due to pyvista implementation, +1 for consistent point numbers
    # almost uniform grid: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    full_sphere = pv.Sphere(theta_resolution=n+1,phi_resolution=n+1)
    ZZ, Z, MZ = manifold_to_GM(full_sphere)
    MZ = pairwise_spherical_distances(full_sphere.points)
    MZ = MZ/MZ.max()
    ZZ.g = MZ.astype(np.float32)
    return full_sphere, ZZ, Z, MZ

def create_torus(n):
    torus = pv.ParametricTorus(u_res=n+1, v_res=n+1, w_res=n+1)
    ZZ, Z, MZ = manifold_to_GM(torus)
    return torus, ZZ, Z, MZ


def create_plane(n, box_len=None):
    plane = pv.Plane(i_resolution=n,j_resolution=n)
    ZZ, Z, MZ = manifold_to_GM(plane)
    MZ = pairwise_euclidean_distances(plane.points)
    MZ = MZ/MZ.max()
    ZZ.g = MZ
    return plane, ZZ, Z, MZ
    
def create_gaussian_grid(vmin=0., vmax=1., meangrid_num=1., variances = [.5, 1.], correlations = [-.5, 0., .5], var=.1, iso=False):
    mean_x = np.linspace(vmin, vmax, meangrid_num, endpoint=True)  # Range for x-coordinate of mean
    mean_y = np.linspace(vmin, vmax, meangrid_num, endpoint=True)  # Range for y-coordinate of mean
    
    # Create the Gaussian grid
    gaussians = []
    for mx in mean_x:
        for my in mean_y:
            if not iso:
                for var_x in variances:
                    for var_y in variances:
                        for rho in correlations:
                            cov = var * np.array([
                                [var_x, rho * np.sqrt(var_x * var_y)],
                                [rho * np.sqrt(var_x * var_y), var_y],
                            ])
                            mean = np.array([mx, my])
                            gaussians.append((mean, cov))
            else:
                for var_both in variances:
                    var_x = var_both
                    var_y = var_both
                    for rho in correlations:
                        cov = var * np.array([
                            [var_x, rho * np.sqrt(var_x * var_y)],
                            [rho * np.sqrt(var_x * var_y), var_y],
                        ])
                        mean = np.array([mx, my])
                        gaussians.append((mean, cov))
    return gaussians

def calc_BW_distance_matrix(gaussians):
    # Compute distances between all pairs in the grid
    distances = np.zeros((len(gaussians), len(gaussians)))
    for i, (mean1, cov1) in enumerate(gaussians):
        for j, (mean2, cov2) in enumerate(gaussians):
            distances[i, j] = ot.gaussian.bures_wasserstein_distance(mean1, mean2, cov1, cov2) # sqrt ? 
    return distances
    
def create_BW_manifold(n, var=.1):
    BW_grid = create_gaussian_grid(meangrid_num=n, var=var)
    MZ = calc_BW_distance_matrix(BW_grid)
    ZZ = GM(X=BW_grid, mode="gauge_only", g=MZ,normalize_gauge=False,squared=False)
    return None, ZZ, None, ZZ.g


def copy_space(X, square=False):
    ZZ = deepcopy(X)
    if square:
        ZZ.g = ZZ.g**2
    return None, ZZ, None, ZZ.g
    

def Wrapper_REW(X1, X2, lambda_GW=[1e3],eps=1e-3, Z_name="Plane",
                              n_its=40, n=50, thres=1e-5, max_len=1, seed=42, var=.1):
    """
    X1, X2: GM Spaces
    Z_name: Types (str) of Embedding Spaces ("Plane", "Sphere", "Torus", "Circle", "X1", "X2", "BW" (Bures-Wasserstein))
    lambda_GW, eps: float hyperparameters
    n_its: iteration steps
    n: grid points in each dim
    box_len: grid size for Euclidean plane
    thres: Thresholding for exclusion of support points
    var: Baseline variance on BW manifold
    
    Defines Z and estimates REW on Z
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    d_quality = {}
    iternum = 0
    #fix support of Z
    valid_Z = ["Plane", "Sphere", "Torus", "Circle", "X1", "X2", "BW"]
    if Z_name == "Plane":
        m, ZZ, Z, MZ = create_plane(n, 1.)
    elif Z_name == "Sphere":
        m,  ZZ, Z, MZ = create_spherical_space(n)
    elif Z_name == "Torus":
        m, ZZ, Z, MZ = create_torus(n)
    elif Z_name == "BW":
        m, ZZ, Z, MZ = create_BW_manifold(n, var=var)
    elif Z_name == "X1":
        m, ZZ, Z, MZ = copy_space(X1)
    elif Z_name == "X2":
        m, ZZ, Z, MZ = copy_space(X2)
    elif Z_name == "Circle":
        m, ZZ, Z, MZ = create_circular_space(n)
    elif Z_name not in valid_Z:
        print(f"{Z_name} is not valid. Valid list is..")
        print(valid_Z)
    if max_len is not None:
        ZZ.g = max_len * ZZ.g/ZZ.g.max()
        MZ = max_len * MZ/MZ.max()
        
    stable = True
    print("---Parameter Setting---")
    print("eps: ", eps, "lambda: ", lambda_GW, "Domain ", Z_name)
    gamma_X1Y1,gamma_Y2X2,zeta1,zeta2 = REW(X1,X2,Z,n_its = n_its,lambda_GW = lambda_GW,eps = eps, MZ=MZ)

    if not np.isnan(zeta1.mean()) or np.isnan(zeta2.mean()):
        return m, ZZ, Z, gamma_X1Y1, gamma_Y2X2, zeta1, zeta2
    else:
        print("Numerically instable, probably small eps. Returning Nones!")
        return None, None, None, None, None, None, None

def euclidean_barycenter_projection(gamma, Z):
    return np.matmul(gamma.numpy() * gamma.shape[0], Z)

def calculate_foscttm(distance_matrix):
    """
    Calculate the FOSCTTM metric based on a square distance matrix.

    Parameters:
    - distance_matrix (np.ndarray): A square numpy array where the entry (i, j) represents 
      the distance between sample i in dataset A and sample j in dataset B.

    Returns:
    - float: The FOSCTTM value.
    """
    n_samples = distance_matrix.shape[0]
    foscttm_values = []

    for i in range(n_samples):
        # Get distances for row i and sort them
        sorted_indices = np.argsort(distance_matrix[i, :])

        # Find the position of the true match (column i) in the sorted list
        true_match_position = np.where(sorted_indices == i)[0][0]

        # Fraction of samples closer than the true match
        fraction_closer = true_match_position / (n_samples - 1)
        foscttm_values.append(fraction_closer)

    # Average the FOSCTTM values over all samples
    foscttm = np.mean(foscttm_values)

    return foscttm
