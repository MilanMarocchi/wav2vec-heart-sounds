"""Command-line entry point for the mPCG Wav2Vec pipeline.

    mpcg-wav2vec gen-train    ...   # train DiffWave / WaveGrad
    mpcg-wav2vec gen-sample   ...   # generate a synthetic dataset from a trained generator
    mpcg-wav2vec classify-cinc ...  # single-PCG / PCG+ECG (Training-A) ablation
    mpcg-wav2vec classify-vest ...  # multichannel vest ablation
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch


@click.group(context_settings={"show_default": True})
def cli() -> None:
    """mPCG Wav2Vec: synthetic augmentation + heart-sound classification."""


# --- data preparation -----------------------------------------------------

@cli.command("make-splits")
@click.option("--data-dir", "data_dirs", multiple=True, required=True,
              help="directory containing a CinC-style REFERENCE.csv (repeatable to combine databases)")
@click.option("--out", "out_path", required=True, help="output reference/split CSV path")
@click.option("--folds", type=int, default=5)
@click.option("--train", type=float, default=0.6)
@click.option("--valid", type=float, default=0.2)
@click.option("--test", type=float, default=0.2)
@click.option("--seed", type=int, default=42)
def make_splits_cmd(data_dirs, out_path, folds, train, valid, test, seed):
    """Generate a patient-level, label-stratified train/valid/test split CSV."""
    from .datasets.splits import SplitRatios, make_splits_from_dirs, write_splits
    df = make_splits_from_dirs(list(data_dirs), folds=folds,
                               ratios=SplitRatios(train, valid, test), seed=seed)
    path = write_splits(df, out_path)
    counts = {c: df[c].value_counts().to_dict() for c in df.columns if c.startswith("split")}
    click.echo(f"Wrote {len(df)} records x {folds} fold(s) to {path}")
    click.echo(json.dumps(counts, indent=2, default=str))


@cli.command("summarize")
@click.argument("results_json")
@click.option("--group-by", default="run_label", help="comma-separated config fields to group by")
@click.option("--metrics", default="accuracy,uar,sensitivity,specificity,mcc",
              help="comma-separated metric names to show")
@click.option("--out", "out_path", default=None, help="write the Markdown table to this file")
def summarize_cmd(results_json, group_by, metrics, out_path):
    """Aggregate an ablation results JSON into a mean/std Markdown table."""
    from .reporting import load_results, summarize, to_markdown
    summary = summarize(load_results(results_json), group_by=[g.strip() for g in group_by.split(",")])
    table = to_markdown(summary, metrics=[m.strip() for m in metrics.split(",")])
    if out_path:
        Path(out_path).write_text(table + "\n")
        click.echo(f"Wrote summary table to {out_path}")
    click.echo(table)


# --- generative -----------------------------------------------------------

@cli.command("gen-train")
@click.option("--model", "model_name", type=click.Choice(["diffwave", "wavegrad"]), required=True)
@click.option("--data-dir", required=True)
@click.option("--csv", "csv_path", required=True)
@click.option("--output-dir", required=True)
@click.option("--epochs", type=int, default=100)
@click.option("--num-classes", type=int, default=2)
@click.option("--condition-on-ecg", is_flag=True, default=False)
@click.option("--segment-dir", default=None,
              help="cardiac-cycle segmentation dir (enables heart-cycle rearranging)")
@click.option("--rearrange/--no-rearrange", "rearrange_cycles", default=True,
              help="rearrange heart cycles during training (needs --segment-dir)")
@click.option("--prob-contiguous", type=float, default=0.0,
              help="probability of a rotated-contiguous rebuild vs shuffled cycles")
@click.option("--fp16", is_flag=True, default=False)
@click.option("--weights", default="", help="checkpoint to resume from")
@click.option("--logdir", default=None, help="TensorBoard log directory")
@click.option("--max-train-batches", type=int, default=None)
def gen_train(model_name, data_dir, csv_path, output_dir, epochs, num_classes,
              condition_on_ecg, segment_dir, rearrange_cycles, prob_contiguous,
              fp16, weights, logdir, max_train_batches):
    """Train a diffusion generator on CinC records."""
    from .config import get_device
    from .datasets.generative import cinc_generative_dataset
    from .generative import GenerativeTrainer, get_spec

    device = get_device()
    spec = get_spec(model_name)
    model = spec.build_model(num_classes).to(device)
    signal = "ecg" if condition_on_ecg else "pcg"
    dataset = cinc_generative_dataset(
        data_dir, csv_path, "train", fs=spec.sample_rate, mel=spec.mel(signal),
        crop_frames=spec.crop_frames, hop_length=spec.hop_length,
        condition_on_ecg=condition_on_ecg, segment_dir=segment_dir,
        rearrange_cycles=rearrange_cycles, prob_contiguous=prob_contiguous,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    trainer = GenerativeTrainer(model, optimizer, spec.loss, device, output_dir, fp16=fp16,
                                log_dir=logdir, sampler=spec.sample)
    if weights:
        trainer.restore(weights)
    trainer.train(loader, epochs, max_train_batches=max_train_batches)
    click.echo(f"Saved generator to {output_dir}/weights.pt")


@cli.command("gen-sample")
@click.option("--model", "model_name", type=click.Choice(["diffwave", "wavegrad"]), required=True)
@click.option("--weights", required=True)
@click.option("--data-dir", required=True)
@click.option("--csv", "csv_path", required=True)
@click.option("--output-dir", required=True)
@click.option("--num-classes", type=int, default=2)
@click.option("--per-item", type=int, default=1)
@click.option("--fast/--no-fast", default=True, help="fast sampling (DiffWave)")
def gen_sample(model_name, weights, data_dir, csv_path, output_dir, num_classes, per_item, fast):
    """Generate a synthetic dataset from a trained generator."""
    from .config import get_device
    from .datasets.generative import cinc_generative_dataset
    from .generative import GenerativeTrainer, generate_dataset, get_spec

    device = get_device()
    spec = get_spec(model_name)
    model = spec.build_model(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    GenerativeTrainer(model, optimizer, spec.loss, device, output_dir).restore(weights)
    dataset = cinc_generative_dataset(
        data_dir, csv_path, "all", fs=spec.sample_rate, mel=spec.mel("pcg"),
        crop_frames=spec.crop_frames, hop_length=spec.hop_length,
    )
    kwargs = {"fast": fast} if model_name == "diffwave" else {}
    path = generate_dataset(model, spec, dataset, output_dir, device=device,
                            per_item=per_item, sampler_kwargs=kwargs)
    click.echo(f"Wrote manifest {path}")


# --- classification -------------------------------------------------------

@cli.command("classify-cinc")
@click.option("--data-dir", required=True)
@click.option("--csv", "csv_path", required=True)
@click.option("--mode", type=click.Choice(["pcg", "ecg", "pcg_ecg"]), default="pcg")
@click.option("--dataset", default="training-a")
@click.option("--fs", type=int, default=4125)
@click.option("--window-s", type=float, default=4.0)
@click.option("--epochs", type=int, default=20)
@click.option("--augment/--no-augment", default=True)
@click.option("--augment-num", type=int, default=15, help="augmented full-record copies per subject (balanced)")
@click.option("--random-init", is_flag=True, default=False)
@click.option("--reference-train-rnn", is_flag=True, default=False,
              help="legacy regime: half epochs + augmented validation set")
@click.option("--fold", type=int, default=1)
@click.option("--max-batches", type=int, default=None)
@click.option("--results-json", default=None)
@click.option("--logdir", "log_dir", default=None, help="TensorBoard log directory")
def classify_cinc(**kwargs):
    """Run a single-PCG / PCG+ECG classification ablation."""
    from .experiments import cinc
    record = cinc.run(kwargs.pop("data_dir"), kwargs.pop("csv_path"), **kwargs)
    click.echo(json.dumps(record, indent=2, default=str))


@cli.command("classify-vest")
@click.option("--data-dir", required=True)
@click.option("--csv", "csv_path", required=True)
@click.option("--channels", default="1,2,3,4,5,6")
@click.option("--fs", type=int, default=4125)
@click.option("--window-s", type=float, default=2.0)
@click.option("--epochs", type=int, default=20)
@click.option("--augment/--no-augment", default=True)
@click.option("--random-init", is_flag=True, default=False)
@click.option("--lora/--no-lora", default=True)
@click.option("--freeze-encoder", is_flag=True, default=False)
@click.option("--fit-svm/--no-svm", default=True)
@click.option("--loss", type=click.Choice(["ce", "contrastive-focal"]), default="ce",
              help="classifier loss (default cross-entropy)")
@click.option("--fold", type=int, default=1)
@click.option("--max-batches", type=int, default=None)
@click.option("--results-json", default=None)
@click.option("--logdir", "log_dir", default=None, help="TensorBoard log directory")
def classify_vest(data_dir, csv_path, channels, **kwargs):
    """Run a multichannel vest classification ablation."""
    from .experiments import multichannel
    chan_list = [int(c) for c in channels.split(",")]
    record = multichannel.run(data_dir, csv_path, channels=chan_list, **kwargs)
    click.echo(json.dumps(record, indent=2, default=str))


@cli.command("classify-synthetic")
@click.option("--schedule", "schedule_path", required=True,
              help="schedule JSON mixing real + generated (WaveGrad/DiffWave) data")
@click.option("--fs", type=int, default=4125)
@click.option("--window-s", type=float, default=4.0)
@click.option("--random-init", is_flag=True, default=False)
@click.option("--max-batches", type=int, default=None)
@click.option("--results-json", default=None)
@click.option("--logdir", "log_dir", default=None, help="TensorBoard log directory")
def classify_synthetic(schedule_path, **kwargs):
    """Train single-channel PCG through a synthetic-augmentation schedule."""
    from .experiments import synthetic
    record = synthetic.run(schedule_path, **kwargs)
    click.echo(json.dumps(record, indent=2, default=str))


@cli.command("classify-lsdo")
@click.option("--db", "dbs", multiple=True, required=True,
              help="repeatable NAME:DATA_DIR:CSV entry, one per CinC database")
@click.option("--holdout", required=True, help="database name to hold out for testing")
@click.option("--fs", type=int, default=4125)
@click.option("--epochs", type=int, default=20)
@click.option("--augment/--no-augment", default=True)
@click.option("--random-init", is_flag=True, default=False)
@click.option("--reference-train-rnn", is_flag=True, default=False,
              help="legacy regime: half epochs + augmented validation set")
@click.option("--max-batches", type=int, default=None)
@click.option("--results-json", default=None)
def classify_lsdo(dbs, holdout, **kwargs):
    """Leave-source-database-out: train on all but one CinC database, test on the held-out one."""
    from .experiments import cinc
    databases = {}
    for entry in dbs:
        name, data_dir, csv_path = entry.split(":", 2)
        databases[name] = (data_dir, csv_path)
    record = cinc.run_leave_out_db(databases, holdout, **kwargs)
    click.echo(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    cli()
