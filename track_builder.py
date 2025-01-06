'''
Track building script.
Separated the track code for modularity and reusability.

Sourced from:
@ highway-env (An Environment for Autonomous Driving Decision-Making)
Edouard Leurent - 2018
https://github.com/eleurent/highway-env
'''


import numpy as np
from highway_env.road.lane import CircularLane, LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork


def make_road(np_random, show_trajectories=False) -> Road:
    net = RoadNetwork()

    # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
    speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

    # Initialise First Lane
    lane = StraightLane(
        [42, 0],
        [100, 0],
        line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        width=5,
        speed_limit=speedlimits[1],
    )

    # Add Lanes to Road Network - Straight Section
    net.add_lane("a", "b", lane)
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [42, 5],
            [100, 5],
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            width=5,
            speed_limit=speedlimits[1],
        ),
    )

    # 2 - Circular Arc #1
    center1 = [100, -20]
    radii1 = 20
    net.add_lane(
        "b",
        "c",
        CircularLane(
            center1,
            radii1,
            np.deg2rad(90),
            np.deg2rad(-1),
            width=5,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speedlimits[2],
        ),
    )
    net.add_lane(
        "b",
        "c",
        CircularLane(
            center1,
            radii1 + 5,
            np.deg2rad(90),
            np.deg2rad(-1),
            width=5,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speedlimits[2],
        ),
    )

    # 3 - Vertical Straight
    net.add_lane(
        "c",
        "d",
        StraightLane(
            [120, -20],
            [120, -30],
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            width=5,
            speed_limit=speedlimits[3],
        ),
    )
    net.add_lane(
        "c",
        "d",
        StraightLane(
            [125, -20],
            [125, -30],
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            width=5,
            speed_limit=speedlimits[3],
        ),
    )

    # 4 - Circular Arc #2
    center2 = [105, -30]
    radii2 = 15
    net.add_lane(
        "d",
        "e",
        CircularLane(
            center2,
            radii2,
            np.deg2rad(0),
            np.deg2rad(-181),
            width=5,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speedlimits[4],
        ),
    )
    net.add_lane(
        "d",
        "e",
        CircularLane(
            center2,
            radii2 + 5,
            np.deg2rad(0),
            np.deg2rad(-181),
            width=5,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speedlimits[4],
        ),
    )

    # 5 - Circular Arc #3
    center3 = [70, -30]
    radii3 = 15
    net.add_lane(
        "e",
        "f",
        CircularLane(
            center3,
            radii3 + 5,
            np.deg2rad(0),
            np.deg2rad(136),
            width=5,
            clockwise=True,
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            speed_limit=speedlimits[5],
        ),
    )
    net.add_lane(
        "e",
        "f",
        CircularLane(
            center3,
            radii3,
            np.deg2rad(0),
            np.deg2rad(137),
            width=5,
            clockwise=True,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=speedlimits[5],
        ),
    )

    # 6 - Slant
    net.add_lane(
        "f",
        "g",
        StraightLane(
            [55.7, -15.7],
            [35.7, -35.7],
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            width=5,
            speed_limit=speedlimits[6],
        ),
    )
    net.add_lane(
        "f",
        "g",
        StraightLane(
            [59.3934, -19.2],
            [39.3934, -39.2],
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            width=5,
            speed_limit=speedlimits[6],
        ),
    )

    # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
    center4 = [18.1, -18.1]
    radii4 = 25
    net.add_lane(
        "g",
        "h",
        CircularLane(
            center4,
            radii4,
            np.deg2rad(315),
            np.deg2rad(170),
            width=5,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speedlimits[7],
        ),
    )
    net.add_lane(
        "g",
        "h",
        CircularLane(
            center4,
            radii4 + 5,
            np.deg2rad(315),
            np.deg2rad(165),
            width=5,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speedlimits[7],
        ),
    )
    net.add_lane(
        "h",
        "i",
        CircularLane(
            center4,
            radii4,
            np.deg2rad(170),
            np.deg2rad(56),
            width=5,
            clockwise=False,
            line_types=(LineType.CONTINUOUS, LineType.NONE),
            speed_limit=speedlimits[7],
        ),
    )
    net.add_lane(
        "h",
        "i",
        CircularLane(
            center4,
            radii4 + 5,
            np.deg2rad(170),
            np.deg2rad(58),
            width=5,
            clockwise=False,
            line_types=(LineType.STRIPED, LineType.CONTINUOUS),
            speed_limit=speedlimits[7],
        ),
    )

    # 8 - Circular Arc #5 - Reconnects to Start
    center5 = [43.2, 23.4]
    radii5 = 18.5
    net.add_lane(
        "i",
        "a",
        CircularLane(
            center5,
            radii5 + 5,
            np.deg2rad(240),
            np.deg2rad(270),
            width=5,
            clockwise=True,
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            speed_limit=speedlimits[8],
        ),
    )
    net.add_lane(
        "i",
        "a",
        CircularLane(
            center5,
            radii5,
            np.deg2rad(238),
            np.deg2rad(268),
            width=5,
            clockwise=True,
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            speed_limit=speedlimits[8],
        ),
    )

    road = Road(
        network=net,
        np_random=np_random,
        record_history=show_trajectories,
    )
    return road