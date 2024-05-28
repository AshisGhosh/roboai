import omni.replicator.core as rep

with rep.new_layer():
    rep.settings.carb_settings("/omni/replicator/RTSubframes", 8)
    
    distance_light = rep.create.light(rotation=(315,0,0), intensity=3000, light_type="distant")

    ENV = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Environments/Simple_Room/simple_room.usd'
    CRACKER_BOX = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd'
    SUGAR_BOX = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd'
    SOUP_CAN = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd'
    MUSTARD_BOTTLE = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd'

    
    def randomize_objects():
        cracker_box = rep.create.from_usd(CRACKER_BOX, semantics=[('class', 'cracker box')])
        cracker_plane_samp = rep.create.plane(scale=1, position=(0, 0, 0.1), rotation=(90, 0, 0), visible=False)

        sugar_box = rep.create.from_usd(SUGAR_BOX, semantics=[('class', 'sugar box')])
        sugar_plane_samp = rep.create.plane(scale=1, position=(0, 0, 0.085), rotation=(90, 0, 0), visible=False)

        soup_can = rep.create.from_usd(SOUP_CAN, semantics=[('class', 'soup can')])
        soup_plane_samp = rep.create.plane(scale=1, position=(0, 0, 0.05), rotation=(90, 0, 0), visible=False)

        mustard_bottle = rep.create.from_usd(MUSTARD_BOTTLE, semantics=[('class', 'mustard bottle')])
        mustard_plane_samp = rep.create.plane(scale=1, position=(0, 0, 0.095), rotation=(90, 0, 0), visible=False)       
    
        with rep.utils.sequential():
            with cracker_box:
                rep.modify.pose(rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)))
                rep.randomizer.scatter_2d(
                    cracker_plane_samp,
                    seed=1,
                    no_coll_prims=[],
                    check_for_collisions=True
                    )
            with sugar_box:
                rep.modify.pose(rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)))
                rep.randomizer.scatter_2d(
                    sugar_plane_samp,
                    seed=2,
                    no_coll_prims=[rep.get.prims(path_match='03_cracker_box', prim_types=['Mesh'])],
                    check_for_collisions=True
                    )
            with soup_can:
                rep.modify.pose(rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)))
                rep.randomizer.scatter_2d(
                    soup_plane_samp,
                    seed=3,
                    no_coll_prims=[
                        rep.get.prims(path_match='03_cracker_box', prim_types=['Mesh']),
                        rep.get.prims(path_match='04_sugar_box', prim_types=['Mesh'])
                        ],
                    check_for_collisions=True
                    )
            with mustard_bottle:
                rep.modify.pose(rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)))
                rep.randomizer.scatter_2d(
                    mustard_plane_samp,
                    seed=4,
                    no_coll_prims=[
                        rep.get.prims(path_match='03_cracker_box', prim_types=['Mesh']),
                        rep.get.prims(path_match='04_sugar_box', prim_types=['Mesh']),
                        rep.get.prims(path_match='05_tomato_soup_can', prim_types=['Mesh']),
                        ],
                    check_for_collisions=True
                    )
        return [cracker_box.node, sugar_box.node, soup_can.node, mustard_bottle.node]
        
    
    # Register randomization
    rep.randomizer.register(randomize_objects)
    
    # Setup the static elements
    env = rep.create.from_usd(ENV)
    
    # Setup camera and attach it to render product
    camera = rep.create.camera(
        focal_length=24.0,
        position=(0, 1.6, 0.9),
        rotation=(60, 0, 180)
    )
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir="_output", semantic_segmentation=True, colorize_instance_segmentation=True, rgb=True, )
    writer.attach([render_product])

    with rep.trigger.on_frame(num_frames=200):
        rep.randomizer.randomize_objects()