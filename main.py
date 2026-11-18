"""Command-line entry point.

    python main.py explain     # one training step with every calculation printed
    python main.py train       # train, logging every iteration, then save checkpoint + results
    python main.py generate    # sample text from a saved checkpoint
    python main.py plot        # draw loss curves, attention maps, ... from saved results
"""

import argparse
import glob
import os
import sys
from dataclasses import asdict, fields

import torch

from minigpt import (MODEL_TYPES, PRESETS, CharTokenizer, Checkpoint, ConsoleLog, Corpus, GPTConfig,
                     LanguageModel, TextDataset, TextGenerator, Tracer, Trainer, TrainingHistory,
                     Walkthrough)


class GPTApp:
    """Parses the command line and runs explain / train / generate / plot."""

    def __init__(self):
        """Build the argument parser once, so `run` can be called repeatedly (e.g. in tests)."""
        self.parser = self._build_parser()

    def run(self, argv=None):
        """Parse arguments and dispatch to the chosen command.

        Args:
            argv (list[str], optional): Arguments to parse; defaults to sys.argv[1:].
        """
        args = self.parser.parse_args(argv)
        sys.stdout.reconfigure(line_buffering=True)  # show every log line immediately, even when piped
        getattr(self, args.command)(args)

    # -- commands --------------------------------------------------------------

    def explain(self, args):
        """Run the Walkthrough: one fully traced training step plus traced generation.

        Args:
            args (argparse.Namespace): Parsed `explain` arguments.
        """
        tokenizer, text, cfg = self._setup(args)
        tracer = Tracer(max_rows=args.rows, max_cols=args.cols,
                        detail_layers=args.detail_layers, detail_heads=args.detail_heads)
        Walkthrough(text, tokenizer, cfg, args.model, tracer).run(generate_tokens=args.generate_tokens)

    def train(self, args):
        """Train a model, then save the checkpoint, the metrics history, the log and a text sample.

        Args:
            args (argparse.Namespace): Parsed `train` arguments.
        """
        tokenizer, text, cfg = self._setup(args)
        results_dir = args.results_dir or f"results/{args.model}-{args.preset}"
        with ConsoleLog(os.path.join(results_dir, "train.log")):
            tracer = Tracer(detail_layers=args.detail_layers, detail_heads=args.detail_heads)
            dataset = TextDataset(text, tokenizer, cfg, tracer)
            model = LanguageModel.create(args.model, cfg, tracer).to(cfg.device)
            print(f"{type(model).__name__}: {model.num_parameters() / 1e6:.3f} M parameters | "
                  f"preset '{args.preset}' | device {cfg.device}")

            history = TrainingHistory(model=args.model, preset=args.preset, config=asdict(cfg))
            try:
                Trainer(model, dataset, cfg, tracer, history).train(
                    log_every=args.log_every, trace_every=args.trace_every)
            except KeyboardInterrupt:
                print("\nInterrupted — saving the model trained so far.")

            checkpoint = args.checkpoint or f"checkpoints/{args.model}.pt"
            Checkpoint.save(checkpoint, model, args.model, cfg, tokenizer)
            history.save(os.path.join(results_dir, "history.json"))
            print(f"saved checkpoint → {checkpoint} | metrics → {results_dir}/history.json "
                  f"| total time {history.total_seconds / 60:.1f} min")
            if args.sample_tokens:
                sample = TextGenerator(model, tokenizer, cfg.device).generate(max_new_tokens=args.sample_tokens)
                print(sample)
                with open(os.path.join(results_dir, "sample.txt"), "w", encoding="utf-8") as f:
                    f.write(sample)

    def generate(self, args):
        """Load a checkpoint and print generated text.

        Args:
            args (argparse.Namespace): Parsed `generate` arguments.
        """
        torch.manual_seed(args.seed)
        model, tokenizer, cfg = Checkpoint.load(args.checkpoint, Tracer(), device=args.device)
        text = TextGenerator(model, tokenizer, cfg.device).generate(
            args.prompt, args.max_new_tokens, trace_tokens=args.trace_tokens)
        print(text)

    def plot(self, args):
        """Draw every chart from saved histories and a checkpoint.

        Args:
            args (argparse.Namespace): Parsed `plot` arguments.
        """
        from minigpt.plots import TrainingPlots  # matplotlib is only needed for this command

        paths = args.runs or sorted(glob.glob("results/*/history.json"))
        histories = [TrainingHistory.load(p) for p in paths]
        model, tokenizer, _ = Checkpoint.load(args.checkpoint, Tracer(), device="cpu")
        for path in TrainingPlots(args.out_dir).make_all(histories, model, tokenizer, args.prompt):
            print(f"saved {path}")

    # -- helpers ---------------------------------------------------------------

    def _setup(self, args):
        """Load the text, build the tokenizer and the config, and seed the RNG.

        Args:
            args (argparse.Namespace): Parsed arguments containing --data, --preset and overrides.

        Returns:
            tuple[CharTokenizer, str, GPTConfig]: (tokenizer, full text, config).
        """
        text = Corpus(args.data).text
        tokenizer = CharTokenizer.from_text(text)
        overrides = {f.name: getattr(args, f.name) for f in fields(GPTConfig) if f.name != "vocab_size"}
        cfg = GPTConfig.from_preset(args.preset, vocab_size=tokenizer.vocab_size, **overrides)
        torch.manual_seed(cfg.seed)
        return tokenizer, text, cfg

    def _build_parser(self):
        """Define every command and its flags.

        Returns:
            argparse.ArgumentParser: The configured parser.
        """
        parser = argparse.ArgumentParser(description="A GPT from scratch, with every calculation visible.")
        commands = parser.add_subparsers(dest="command", required=True)

        explain = commands.add_parser("explain", help="trace one training step + generation in full detail")
        self._add_model_args(explain, default_preset="tiny")
        self._add_trace_args(explain)
        explain.add_argument("--rows", type=int, default=8, help="matrix rows to print per tensor")
        explain.add_argument("--cols", type=int, default=8, help="matrix columns to print per tensor")
        explain.add_argument("--generate-tokens", type=int, default=3, help="sampling steps to trace")

        train = commands.add_parser("train", help="train a model and log every iteration")
        self._add_model_args(train, default_preset="small")
        self._add_trace_args(train)
        train.add_argument("--log-every", type=int, default=1, help="print the loss every N iterations")
        train.add_argument("--trace-every", type=int, default=0,
                           help="print a full calculation trace every N iterations (0 = never)")
        train.add_argument("--checkpoint", default=None, help="where to save (default checkpoints/<model>.pt)")
        train.add_argument("--results-dir", default=None,
                           help="where to save log, metrics and sample (default results/<model>-<preset>)")
        train.add_argument("--sample-tokens", type=int, default=500, help="characters to generate after training")

        generate = commands.add_parser("generate", help="sample text from a trained checkpoint")
        generate.add_argument("--checkpoint", default="checkpoints/gpt.pt")
        generate.add_argument("--prompt", default="", help="text to start from (default: a newline)")
        generate.add_argument("--max-new-tokens", type=int, default=500)
        generate.add_argument("--trace-tokens", type=int, default=0,
                              help="show the probabilities behind the first N sampled characters")
        generate.add_argument("--device", default="auto")
        generate.add_argument("--seed", type=int, default=1337)

        plot = commands.add_parser("plot", help="draw charts from saved training results")
        plot.add_argument("--runs", nargs="*", help="history.json files (default: results/*/history.json)")
        plot.add_argument("--checkpoint", default="checkpoints/gpt.pt", help="GPT checkpoint for attention plots")
        plot.add_argument("--prompt", default="ROMEO:\nWhat light", help="text whose attention is drawn")
        plot.add_argument("--out-dir", default="docs/images")
        return parser

    @staticmethod
    def _add_model_args(parser, default_preset):
        """Add --model, --preset, --data and one override flag per GPTConfig field.

        Args:
            parser (argparse.ArgumentParser): Sub-command parser to extend.
            default_preset (str): Preset used when --preset is not given.
        """
        parser.add_argument("--model", choices=sorted(MODEL_TYPES), default="gpt")
        parser.add_argument("--preset", choices=list(PRESETS), default=default_preset)
        parser.add_argument("--data", default="data/input.txt", help="training text file")
        group = parser.add_argument_group("hyperparameter overrides (default: from the preset)")
        for f in fields(GPTConfig):
            if f.name != "vocab_size":
                group.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=type(f.default), default=None)

    @staticmethod
    def _add_trace_args(parser):
        """Add the flags that control how much of the network the tracer prints.

        Args:
            parser (argparse.ArgumentParser): Sub-command parser to extend.
        """
        parser.add_argument("--detail-layers", type=int, default=1, help="blocks to trace in full")
        parser.add_argument("--detail-heads", type=int, default=1, help="attention heads per block to trace in full")


if __name__ == "__main__":
    GPTApp().run()
