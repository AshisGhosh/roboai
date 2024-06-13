import omni.replicator.core as rep

with rep.new_layer():
    rep.settings.carb_settings("/omni/replicator/RTSubframes", 8)
    
    # Add Default Light
    distance_light = rep.create.light(rotation=(315,0,0), intensity=3000, light_type="distant")

    # Define paths for the character, the props, the environment and the surface where the assets will be scattered in.

    ENV = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Environments/Simple_Room/simple_room.usd'
    CRACKER_BOX = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd'
    SUGAR_BOX = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd'
    SOUP_CAN = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd'
    MUSTARD_BOTTLE = 'omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd'

    def cracker_box():
        cracker_box = rep.create.from_usd(CRACKER_BOX, semantics=[('class', 'cracker box')])
        
        with cracker_box:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.5, -0.35, 0.1), (0.5, 0.35, 0.1)),
                rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)),
            )

        return cracker_box

    def sugar_box():
        sugar_box = rep.create.from_usd(SUGAR_BOX, semantics=[('class', 'sugar box')])

        with sugar_box:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.5, -0.35, 0.085), (0.5, 0.35, 0.085)),
                rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)),
            )

        return sugar_box
    
    def soup_can():
        soup_can = rep.create.from_usd(SOUP_CAN, semantics=[('class', 'soup can')])

        with soup_can:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.5, -0.35, 0.05), (0.5, 0.35, 0.05)),
                rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)),
            )
        return soup_can
    
    def mustard_bottle():
        mustard_bottle = rep.create.from_usd(MUSTARD_BOTTLE, semantics=[('class', 'mustard bottle')])

        with mustard_bottle:
            rep.modify.pose(
                position=rep.distribution.uniform((-0.5, -0.35, 0.095), (0.5, 0.35, 0.095)),
                rotation=rep.distribution.uniform((-90, 0, -180), (-90, 0, 180)),
            )
        return mustard_bottle
    
    # Register randomization
    rep.randomizer.register(cracker_box)
    rep.randomizer.register(sugar_box)
    rep.randomizer.register(soup_can)
    rep.randomizer.register(mustard_bottle)

    # Setup the static elements
    env = rep.create.from_usd(ENV)
    # with env:
    #     # Add physics to the collision floor, the props already have rigid-body physics applied
    #     rep.physics.collider()

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

    with rep.trigger.on_frame(num_frames=20):
        rep.randomizer.cracker_box()
        rep.randomizer.sugar_box()
        rep.randomizer.soup_can()
        rep.randomizer.mustard_bottle()