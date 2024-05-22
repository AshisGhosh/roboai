CLEAR_TABLE_PLAN = {
    "clear the table":
        """
        Given the objects: [object1, object2, object3]
        1. Pick up object 1
        2. Place object 1
        3. Pick up object 2
        4. Place object 2
        5. Pick up object 3
        6. Place object 3
    """
    }

CLEAN_BATHROOM_PLAN = {
    "clean the bathroom":
        """
        Given the bathroom is dirty and has a toilet, sink, and shower
        1. Spray the shower with cleaner
        2. Spray the sink with cleaner
        3. Spray the toilet with cleaner
        4. Scrub the sink
        5. Scrub the toilet
        6. Scrub the shower
        """
    }


PLANS = {
    **CLEAR_TABLE_PLAN,
    **CLEAN_BATHROOM_PLAN,
}
