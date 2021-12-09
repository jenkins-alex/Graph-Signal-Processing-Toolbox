import plotly.graph_objects as go
import numpy as np
from sklearn.datasets import make_swiss_roll
from manifolds.spectral_clustering import SpectralClustering

class GraphVisualiser:

    def __init__(self, data, edges=None, node_colors='aliceblue', 
                 edge_colors='azure', dimension='3d', draw_bg=False, title=""):
        """ constructor for graph visualiser

        Args:
            data (np.array): (nodes x coordinates) node coordinates
            edges (nx.EdgeView): [graph edges]. Defaults to None.
            node_colors (np.array, optional): [colors to assign to nodes (nodes x 1)]. Defaults to 'aliceblue'.
            edge_colors (color, optional): [colors to assign to edges]. Defaults to 'azure'.
            dimension (str, optional): [dimension for visualisation ['1d', '2d', '3d']]. Defaults to '3d'.
            draw_bg (bool, optional): [whether to draw background axes for plot]. Defaults to False.
            title (str, option): [title for plot]. Defaults to empty string.
        """
        assert dimension in ['1d', '2d', '3d'], "Dimension should be either '1d', '2d' or '3d'."
        self.data = data
        self.edges = edges
        self.node_colors = node_colors
        self.edge_colors = edge_colors
        self.dimension = dimension
        self.draw_bg = draw_bg
        self.title = title

        if self.dimension == '1d':
            # cast data onto a unit circle
            x = self.data[:, 0]
            x_new, y_new = self._cast_unit_circle(list(x))
            self.data = np.array([x_new, y_new]).T

    def _edge_coordinates(self):
        """ find start and end coordinates of each edge

        Returns:
            tuple of lists: x, y and z coordinates of edge start and end points
        """
        if self.edges is None:
            return None, None, None

        # create lists that contain the starting and ending coordinates of each edge
        x_edges=[]
        y_edges=[]
        z_edges=[]

        for edge in self.edges:
            # fill x coordinates
            x_coords = [self.data[edge[0]][0], self.data[edge[1]][0], None]
            x_edges += x_coords
            
            # fill y coordinates
            y_coords = [self.data[edge[0]][1], self.data[edge[1]][1], None]
            y_edges += y_coords
            if self.dimension != '3d':
                continue
            
            # fill z coordinates
            z_coords = [self.data[edge[0]][2], self.data[edge[1]][2], None]
            z_edges += z_coords

        if self.dimension == '1d':
            return x_edges, y_edges, None

        if self.dimension == '2d':
            return x_edges, y_edges, None

        return x_edges, y_edges, z_edges

    def _cast_unit_circle(self, coords):
        """ cast 1d coordinate data to a unit circle

        Args:
            coords (list or np.array): 1D coordinate data

        Returns:
            [tuple of lists]: x and y coordinates of points on the unit circle 
        """
        theta = self._cast_unit_interval(coords) * 2 * np.pi
        x_new = np.cos(theta)
        y_new = np.sin(theta)
        return list(x_new), list(y_new)

    def _cast_unit_interval(self, coords):
        """ min-max normalisation of numpy array

        Args:
            coords (list or np.array): data to normalise

        Returns:
            list or np.array: normalised data
        """
        return (coords - np.min(coords)) / (np.max(coords) - np.min(coords))

    def _trace_graph_edges(self, x_edges, y_edges, z_edges):
        """ trace graph edges as plotly object

        Args:
            x_edges (list): start and end edge x coordinates
            y_edges (list): start and end edge y coordinates
            z_edges (list): start and end edge z coordinates

        Returns:
            go.Scatter: Scatter object with edges plotted
        """
        if self.dimension == '3d':
            trace_edges = go.Scatter3d(x=x_edges,
                                y=y_edges,
                                z=z_edges,
                                mode='lines',
                                line=dict(color=self.edge_colors, width=.5),
                                hoverinfo='none')
        else: 
            trace_edges = go.Scatter(x=x_edges,
                                y=y_edges,
                                mode='lines',
                                line=dict(color=self.edge_colors, width=.5),
                                hoverinfo='none')
        return trace_edges

    def _trace_colored_nodes(self):
        """ trace coloured nodes as plotly object

        Returns:
            go.Scatter: Scatter object with nodes plotted
        """
        if self.dimension == '3d':
            trace_nodes = go.Scatter3d(x=self.data[:,0],
                                       y=self.data[:,1],
                                       z=self.data[:,2],
                                       marker=dict(symbol='circle',
                                                   size=4,
                                                   color=self.node_colors,
                                                   line=dict(color='black',
                                                             width=0.5)),
                                       mode='markers')
        else:
            trace_nodes = go.Scatter(x=self.data[:,0],
                                     y=self.data[:,1],
                                     marker=dict(symbol='circle',
                                                 size=4,
                                                 color=self.node_colors,
                                                 line=dict(color='black',
                                                           width=0.5)),
                                     mode='markers')
        return trace_nodes

    def _create_graph_axis(self):
        """ graph axis for plotly visualisation

        Returns:
            dict: axis settings 
        """
        axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
        return axis

    def _create_graph_layout(self, axis):
        """ layout for visualisation"""
        if self.draw_bg:
            layout = go.Layout(title=self.title,
                    width=650,
                    height=625,
                    showlegend=False,
                    margin=dict(t=100),
                    hovermode='closest')
        else:
            scene = dict(xaxis=dict(axis), yaxis=dict(axis))
            if self.dimension == '3d':
                scene['zaxis'] = dict(axis)
            
            layout = go.Layout(title=self.title,
                        width=650,
                        height=625,
                        showlegend=False,
                        scene=scene,
                        margin=dict(t=100),
                        hovermode='closest')
        return layout
    
    def visualise(self):
        """ visualise graph data

        Returns:
            fig: plotly figure
        """
        x_edges, y_edges, z_edges = self._edge_coordinates()
        trace_edges = self._trace_graph_edges(x_edges, y_edges, z_edges)
        trace_nodes = self._trace_colored_nodes()
        axis = self._create_graph_axis()
        layout = self._create_graph_layout(axis)
        fig = go.Figure(data=[trace_edges, trace_nodes], layout=layout)
        return fig


def test():
    # create the swiss roll data
    n_samples = 500
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.05)
    X[:, 1] *= 0.5  # make it thinner

    # test out the spectral clustering to define edges and colors
    sc = SpectralClustering(X, norm=False, kernel='rbf', gamma=1/32, edge_thresh=4)
    edges = sc._get_edges()
    w, v = sc.eig_decompose()

    # the first eigenvalue is a maximally smooth constant, can be omitted
    spectral_matrix = v[:, 1:]
    # create a reduced spectral vector, only use first 3 spectral components
    q = spectral_matrix[:, :3]

    # min-max normalise each component of spectral vector to range 0-1
    colors = (q - q.min(axis=0)) / (q.max(axis=0) - q.min(axis=0)) * 255
    colors = colors.astype(int)

    gv = GraphVisualiser(q, node_colors=colors, edges=edges, dimension='1d')
    fig = gv.visualise()
    fig.show()