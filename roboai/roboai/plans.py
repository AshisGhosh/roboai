CLEAR_TABLE_PLAN = {
    "clear the table": """
        1. Navigate to the table
        2. Scan the table for objects
        3. Rollout pick and place plan to remove objects
        """
}

CLEAN_BATHROOM_PLAN = {
    "clean the bathroom": """
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
