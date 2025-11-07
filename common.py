from dataclasses import dataclass
from typing import Iterable, List
import math

# Functions for approximate comparison of floating point numbers
# fmt: off
EPS = 1e-6
def is_eq(x, y): return abs(x - y) <= EPS
def is_lt(x, y): return x - y < -EPS
def is_le(x, y): return x - y <= EPS
def is_gt(x, y): return x - y > EPS
def is_ge(x, y): return x - y >= -EPS
def eps_floor(x): return math.floor(x + EPS)
def eps_ceil(x): return math.ceil(x - EPS)
def eps_round(x): return math.ceil(x - 0.5 + EPS)
def eps_frac(x): return x - eps_floor(x)
def is_integral(x): return eps_frac(x) <= EPS
# fmt: on

# Type aliases
Agent = int
Time = int


@dataclass(frozen=True, order=True)
class XY:
    """A 2D coordinate on the grid. All coordinates are integers and use (x, y) with origin (0, 0)
    at the top-left of the map (x increases to the right, y increases downward).

    Attributes:
        x: X-coordinate (integer).
        y: Y-coordinate (integer).
    """

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x},{self.y})"


@dataclass(frozen=True)
class Vertex:
    """A vertex in the time-expanded graph.

    Attributes:
        xy: An ``XY`` coordinate.
        t:  An integer timestep (``Time``).
    """

    xy: XY
    t: Time

    def __str__(self) -> str:
        return f"({self.xy},{self.t})"


@dataclass(frozen=True)
class UndirectedEdge:
    """An undirected edge in the time-expanded graph, representing traversal in either direction at
    a given time.

    Attributes:
        i: One endpoint coordinate (``XY``).
        j: Other endpoint coordinate (``XY``).
        t: Timestep at which the traversal occurs.
    """

    i: XY
    j: XY
    t: Time

    def __post_init__(self):

        # Normalize ordering so the edge is undirected.
        sort = sorted([self.i, self.j])
        object.__setattr__(self, "i", sort[0])
        object.__setattr__(self, "j", sort[1])

    def __str__(self) -> str:
        return f"({self.i},{self.t},{self.j})"


@dataclass(frozen=True)
class DirectedEdge:
    """A directed edge in the time-expanded graph, representing traversal in one direction at a
    given time

    Attributes:
        i: Start coordinate of the directed transition (``XY``).
        j: End coordinate of the directed transition (``XY``).
        t: Timestep at which the directed move occurs.
    """

    i: XY
    j: XY
    t: Time

    def __str__(self) -> str:
        return f"({self.i},{self.t},{self.j})"


@dataclass(frozen=True)
class Path:
    """A candidate path for an agent.

    Attributes:
        agent: Agent number.
        xys:   List of ``XY`` coordinates visited by the path.
    """

    agent: Agent
    xys: List[XY]

    def length(self) -> int:
        """Return the number of vertices in the path."""

        return len(self.xys) - 1

    def vertices(self) -> Iterable[Vertex]:
        """Yield vertices of the path."""

        for t, xy in enumerate(self.xys):
            yield Vertex(xy, t)

    def edges(self) -> Iterable[DirectedEdge]:
        """Yield edges traversed in the path."""

        for t in range(len(self.xys) - 1):
            i = self.xys[t]
            j = self.xys[t + 1]
            yield DirectedEdge(i, j, t)

    def uses_vertex(self, vertex: Vertex) -> bool:
        """Return True if the path visits ``vertex`` (same xy at same time)."""

        return vertex.t < len(self.xys) and self.xys[vertex.t] == vertex.xy

    def uses_directed_edge(self, edge: DirectedEdge) -> bool:
        """Return True if the path traverses ``edge``."""

        return (
            edge.t < len(self.xys) - 1
            and DirectedEdge(self.xys[edge.t], self.xys[edge.t + 1], edge.t) == edge
        )

    def uses_undirected_edge(self, edge: UndirectedEdge) -> bool:
        """Return True if the path traverses the undirected ``edge`` (in either direction)."""

        return (
            edge.t < len(self.xys) - 1
            and UndirectedEdge(self.xys[edge.t], self.xys[edge.t + 1], edge.t) == edge
        )

    def __eq__(self, other) -> bool:
        """Compares if two paths belong to the same agent and uses the same vertices."""

        if not isinstance(other, Path):
            return False
        return (self.agent == other.agent) and (tuple(self.xys) == tuple(other.xys))

    def __hash__(self) -> int:
        """Hash using agent and vertices."""

        return hash((self.agent, tuple(self.xys)))
