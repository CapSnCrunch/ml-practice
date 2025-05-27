from manim import *
import numpy as np

class Grid11x10x2_3D(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.move_camera(frame_center=ORIGIN, zoom=1)
        cubes = VGroup()
        filled_cube = None
        # Create 11x10x2 grid of cubes
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if (i, j, k) == (1, 1, 1):
                        # Filled yellow cube at (1,1,1)
                        filled_cube = Cube(side_length=1, fill_opacity=0.7, fill_color=YELLOW, stroke_width=0)
                        filled_cube.move_to(np.array([i + 0.5, j + 0.5, k + 0.5]))
                    # Wireframe cube
                    wire_cube = Cube(side_length=1, fill_opacity=0, stroke_color=WHITE, stroke_width=0.2)
                    wire_cube.move_to(np.array([i + 0.5, j + 0.5, k + 0.5]))
                    cubes.add(wire_cube)
        self.begin_ambient_camera_rotation(rate=0)
        self.play(Create(cubes), run_time=1)
        if filled_cube:
            self.play(FadeIn(filled_cube), run_time=1)
        self.wait(2)
        self.play(FadeOut(cubes), FadeOut(filled_cube))
