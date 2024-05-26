SCAN_THE_SCENE = {
    "scan the scene": {
        "symbol": "scan_the_scene",
        "description": """
                        Can scan to scene to retreieve information. 
                        Can also be used to identify objects in the scene.
                        Should already be at the required location.
                        """,
    }
}


PICK_OBJECT = {
    "pick object": {
        "symbol": "pick",
        "description": """
                        Can pick up an object in the scene. 
                        Requires specifying the object.
                        """,
    }
}

PLACE_IN_LOCATION = {
    "place in location": {
        "symbol": "place",
        "description": """
                        Can place an object in the scene.
                        Requires already holding the object.
                        Requires specifying the location.
                        """,
    }
}

NAVIGATE_TO_LOCATION = {
    "navigate to location": {
        "symbol": "navigate_to",
        "description": """
                        Can navigate to a location in the scene.
                        Location can also be specified by an object.
                        """,
    }
}

CALL_SUPPORT = {
    "call support": {
        "symbol": "call_support",
        "description": """
                        Can call support for an issue in the scene.
                        """,
    }
}

UPDATE_PLAN = {
    "update plan": {
        "symbol": "update_plan",
        "description": """
                        Can update the plan to a new one.
                        """,
    }
}

ROLLOUT_PICK_AND_PLACE_PLAN = {
    "rollout pick and place plan": {
        "symbol": "rollout",
        "description": """
                        Given an observation or a scan step, can rollout a pick and place plan.
                        """,
    }
}


SKILLS = {
    **SCAN_THE_SCENE,
    **PICK_OBJECT,
    **PLACE_IN_LOCATION,
    **NAVIGATE_TO_LOCATION,
    **CALL_SUPPORT,
    **UPDATE_PLAN,
    **ROLLOUT_PICK_AND_PLACE_PLAN,
}
