import networkx as nx
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from omnigibson.sensors import VisionSensor


def visualize_ascii_scene_graph(scene, G):
    # def print_graph_ascii(G):
    #     for line in nx.generate_adjlist(G):
    #         print(line)

    # # Example usage:
    # print_graph_ascii(G)
    nx.write_network_text(G)


def visualize_scene_graph(scene, G, show_window=True, realistic_positioning=False):
    """
    Converts the graph into an image and shows it in a cv2 window if preferred.

    Args:
        show_window (bool): Whether a cv2 GUI window containing the visualization should be shown.
        realistic_positioning (bool): Whether nodes should be positioned based on their position in the scene (if True)
            or placed using a graphviz layout (neato) that makes it easier to read edges & find clusters.
    """

    def _draw_graph():
        nodes = list(G.nodes)
        node_labels = {obj: obj.category for obj in nodes}
        # colors = [
        #     "yellow" if obj.category == "agent"
        #     else (
        #         "green" if obj.states.get(object_states.ObjectsInFOVOfRobot, False)
        #         else "red" if object_states.ObjectsInFOVOfRobot in obj.states
        #         else "blue"
        #     )
        #     for obj in nodes
        # ]
        positions = (
            {obj: (-pose[0][1], pose[0][0]) for obj, pose in G.nodes.data("pose")}
            if realistic_positioning
            else nx.nx_pydot.pydot_layout(G, prog="neato")
        )
        nx.drawing.draw_networkx(
            G,
            pos=positions,
            labels=node_labels,
            nodelist=nodes,
            # node_color=colors,
            font_size=4,
            arrowsize=5,
            node_size=150,
        )

        edge_labels = {
            edge: ", ".join(
                f"{state}={value}" for state, value in G.edges[edge]["states"]
            )
            for edge in G.edges
        }
        nx.drawing.draw_networkx_edge_labels(
            G, pos=positions, edge_labels=edge_labels, font_size=4
        )

    # Prepare pyplot figure sized to match the robot video.
    robot = scene.robots[0]
    robot_camera_sensor = next(
        s
        for s in robot.sensors.values()
        if isinstance(s, VisionSensor) and "rgb" in s.modalities
    )
    robot_view = (robot_camera_sensor.get_obs()[0]["rgb"][..., :3]).astype(np.uint8)
    imgheight, imgwidth, _ = robot_view.shape

    figheight = 4.8
    figdpi = imgheight / figheight
    figwidth = imgwidth / figdpi

    # Draw the graph onto the figure.
    fig = plt.figure(figsize=(figwidth, figheight), dpi=figdpi)
    _draw_graph()
    fig.canvas.draw()

    # Convert the canvas to image
    graph_view = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    graph_view = graph_view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    assert graph_view.shape == robot_view.shape
    plt.close(fig)

    # Combine the two images side-by-side
    img = np.hstack((robot_view, graph_view))

    # Convert to BGR for cv2-based viewing.
    if show_window:
        import cv2

        cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("SceneGraph", cv_img)
        cv2.waitKey(1)

    return Image.fromarray(img).save(r"test.png")
