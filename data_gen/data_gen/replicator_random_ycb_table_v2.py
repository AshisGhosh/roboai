import datetime as dt
import omni.replicator.core as rep
import sys
import random

# Boolean switch to exclude training set assets
exclude_finetune_usds = True

finetune_usds = [
    "003_cracker_box.usd",
    "004_sugar_box.usd",
    "005_tomato_soup_can.usd",
    "006_mustard_bottle.usd",
]

with rep.new_layer():
    rep.settings.carb_settings("/omni/replicator/RTSubframes", 12)
    rep.settings.set_stage_up_axis("Y")

    distance_light = rep.create.light(
        rotation=(315, 0, 0), intensity=3000, light_type="distant"
    )

    ENV = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Environments/Simple_Room/simple_room.usd"
    OBJECTS_DIR = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Props/YCB/Axis_Aligned"

    # Retrieve all USD files from the directory
    usd_files = rep.utils.get_usd_files(OBJECTS_DIR, recursive=False)

    # Exclude specific assets if the switch is True
    if exclude_finetune_usds:
        usd_files = [
            file for file in usd_files if file.split("/")[-1] not in finetune_usds
        ]

    # Verify by listing all retrieved USD files
    print("Loaded assets:")
    for usd_file in usd_files:
        print(usd_file)
    sys.stdout.flush()

    objects_data = {
        "002_master_chef_can.usd": {
            "semantic_class": "blue Master Chef coffee bean can",
            "height": 0.070089,
        },
        "003_cracker_box.usd": {
            "semantic_class": "red CheezIt cracker box",
            "height": 0.106719,
        },
        "004_sugar_box.usd": {
            "semantic_class": "yellow and white Domino sugar box",
            "height": 0.088127,
        },
        "005_tomato_soup_can.usd": {
            "semantic_class": "red and white Campbells soup can",
            "height": 0.050928,
        },
        "006_mustard_bottle.usd": {
            "semantic_class": "yellow Frenchs mustard bottle",
            "height": 0.095651,
        },
        "007_tuna_fish_can.usd": {
            "semantic_class": "blue Starkist tuna fish can",
            "height": 0.016769,
        },
        "008_pudding_box.usd": {
            "semantic_class": "black and red Jello pudding box",
            "height": 0.019236,
        },
        "009_gelatin_box.usd": {
            "semantic_class": "red Jello gelatin box",
            "height": 0.014992,
        },
        "010_potted_meat_can.usd": {
            "semantic_class": "blue Spam potted meat tin",
            "height": 0.041772,
        },
        "011_banana.usd": {"semantic_class": "yellow banana fruit", "height": 0.019325},
        "019_pitcher_base.usd": {
            "semantic_class": "blue beverage pitcher",
            "height": 0.121194,
        },
        "021_bleach_cleanser.usd": {
            "semantic_class": "white Soft Scrub bleach bottle",
            "height": 0.125290,
        },
        "024_bowl.usd": {
            "semantic_class": "red enamel coated metal bowl",
            "height": 0.027505,
        },
        "025_mug.usd": {
            "semantic_class": "red enamel coated metal mug",
            "height": 0.040651,
        },
        "035_power_drill.usd": {
            "semantic_class": "orange and black Black and Decker power drill",
            "height": 0.093418,
        },
        "036_wood_block.usd": {
            "semantic_class": "brown wood block",
            "height": 0.103046,
        },
        "037_scissors.usd": {
            "semantic_class": "gray and yellow scissors",
            "height": 0.007936,
        },
        "040_large_marker.usd": {
            "semantic_class": "black Expo dry erase marker",
            "height": 0.009738,
        },
        "051_large_clamp.usd": {
            "semantic_class": "black large spring clamp",
            "height": 0.018206,
        },
        "052_extra_large_clamp.usd": {
            "semantic_class": "black extra large spring clamp",
            "height": 0.018238,
        },
        "061_foam_brick.usd": {"semantic_class": "red foam brick", "height": 0.050928},
    }

    def randomize_objects():
        num_objects = random.randint(1, 6)
        selected_assets = random.sample(usd_files, num_objects)
        print(f"Selected assets: {selected_assets}")
        sys.stdout.flush()

        objects = []
        no_coll_prims = []

        with rep.utils.sequential():
            for i, asset in enumerate(selected_assets):
                asset_name = asset.split("/")[-1]
                object_data = objects_data.get(
                    asset_name, {"semantic_class": "unknown object", "height": 0}
                )
                semantic_class = object_data["semantic_class"]
                height = object_data["height"]
                print(
                    f"Adding asset: {asset_name} with semantic class: {semantic_class} and height: {height}"
                )
                sys.stdout.flush()

                obj = rep.create.from_usd(asset, semantics=[("class", semantic_class)])
                plane_samp = rep.create.plane(
                    scale=1, position=(0, 0, height), rotation=(90, 0, 0), visible=False
                )

                with obj:
                    print(f"Modifying pose for {asset_name}")
                    if asset_name in [
                        "008_pudding_box.usd",
                        "009_gelatin_box.usd",
                        "037_scissors.usd",
                        "040_large_marker.usd",
                        "051_large_clamp.usd",
                        "052_extra_large_clamp.usd",
                    ]:
                        rep.modify.pose(
                            rotation=rep.distribution.uniform(
                                (0, 0, -180), (0, 0, 180), seed=i
                            )
                        )
                    else:
                        rep.modify.pose(
                            rotation=rep.distribution.uniform(
                                (-90, 0, -180), (-90, 0, 180), seed=i
                            )
                        )

                    print(f"Scattering {asset_name}")
                    sys.stdout.flush()
                    try:
                        rep.randomizer.scatter_2d(
                            plane_samp,
                            seed=i,
                            no_coll_prims=no_coll_prims,
                            check_for_collisions=True,
                        )
                        print(f"Successfully scattered {asset_name}")
                    except Exception as e:
                        print(f"Error scattering {asset_name}: {e}")

                no_coll_prims.append(obj.node)
                objects.append(obj.node)
                print(f"Completed adding {asset_name}")
                sys.stdout.flush()

        return objects

    rep.randomizer.register(randomize_objects)

    env = rep.create.from_usd(ENV)

    camera = rep.create.camera(
        focal_length=24.0, position=(0, 1.6, 0.9), rotation=(60, 0, 180)
    )
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=f"/isaac-sim/roboai/shared/data/image_exports/replicator_output/{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        semantic_segmentation=True,
        colorize_instance_segmentation=True,
        rgb=True,
    )
    writer.attach([render_product])

    with rep.trigger.on_frame(num_frames=1000):
        rep.randomizer.randomize_objects()

    rep.orchestrator.run()
