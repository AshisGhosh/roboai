default:
    just -l

train:
    xhost +
    docker compose up model-training --build
    xhost -

eval:
    xhost +
    docker compose build model-training
    docker compose run model-training poetry run python -u -m model_training.eval
    xhost -

convert-dataset:
    xhost +
    docker compose run model-training poetry run python -u -m model_training.data --build
    xhost -

score-responses:
    xhost +
    docker compose build model-training
    # docker compose run model-training poetry run python -u -m shared.scripts.score_responses -m moondream2_responses_20240614_212629.json -g model_training/data/ycb_isaac_raw/gt_responses_20240606_083247.json 
    docker compose run model-training poetry run python -u -m shared.scripts.score_responses -m finetuned_moondream2_responses_20240614_220208.json -g model_training/data/ycb_isaac_raw/gt_responses_20240606_083247.json 
    xhost -

generate-graphs:
    xhost +
    docker compose run model-training poetry run python -u -m shared.scripts.generate_tables_graphs response_scores_vikhyatk_moondream2_20240614_214233.json response_scores_vikhyatk_moondream2_20240614_220404.json
    xhost -


make-owner:
    sudo chown -R 1000:1000 .