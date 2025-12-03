import os
import expyriment
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

STIM_DIR = Path("stimuli")

ANGLES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
N_PAIRS = 1
REPETITIONS = 3

FIXATION_TIME = 500
ISI_TIME = 500


def build_experiment():
    exp = expyriment.design.Experiment(
        name="Mental Rotation 3D"
    )
    expyriment.control.initialize(exp)

    block = expyriment.design.Block(name="main")

    os.makedirs("results", exist_ok=True)

    # Create trials
    for angle in ANGLES:
        for same in [0, 1]:
            for rep in range(REPETITIONS):
                trial = expyriment.design.Trial()

                stim_filename = f"pair1_rot{angle}_same{same}.png"
                stim_path = STIM_DIR / stim_filename

                if not stim_path.exists():
                    raise IOError(f"Missing stimulus file: {stim_path}")

                stim = expyriment.stimuli.Picture(str(stim_path))
                stim.preload()

                trial.add_stimulus(stim)

                trial.set_factor("angle", angle)
                trial.set_factor("same", same)
                trial.set_factor("rep", rep)

                block.add_trial(trial)

    block.shuffle_trials()
    exp.add_block(block)

    exp.add_data_variable_names([
        "angle", "same", "rep", "response_same", "correct", "rt_ms"
    ])

    return exp


def analyze_and_plot(results):
    num_run = len(os.listdir("results/")) // 2
    csv_filename = f"results/mental_rotation_data_{num_run}.csv"
    fig_filename = f"results/mental_rotation_data_{num_run}.png"

    header = ["angle", "same", "rep", "response_same", "correct", "rt_ms"]
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)

    data = np.array(results, dtype=float)

    # same_flag = 1 -> "same" trials, same_flag = 0 -> "different" trials
    def compute_stats_and_regression(sub_data, label):
        unique_angles = np.unique(sub_data[:, 0])
        mean_rt = []
        acc = []

        for angle in unique_angles:
            mask_angle = sub_data[:, 0] == angle
            subset = sub_data[mask_angle]

            # accuracy at this angle
            acc.append(np.mean(subset[:, 4]))

            # mean RT on correct trials only
            correct_subset = subset[subset[:, 4] == 1]
            if len(correct_subset) > 0:
                mean_rt.append(np.mean(correct_subset[:, 5]))
            else:
                mean_rt.append(np.nan)

        unique_angles = unique_angles.astype(float)
        mean_rt = np.array(mean_rt, dtype=float)
        acc = np.array(acc, dtype=float)

        mask_valid = ~np.isnan(mean_rt)
        X = unique_angles[mask_valid]
        Y = mean_rt[mask_valid]

        if len(X) >= 2:
            a, b = np.polyfit(X, Y, deg=1)
            Y_pred = a * X + b

            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            print(f"\n===== Linear Regression RT(theta) — {label} trials =====")
            print(f"RT = {a:.3f} * angle + {b:.3f}")
            print(f"Slope a       = {a:.3f} ms/degree")
            print(f"Intercept b   = {b:.3f} ms")
            print(f"R²            = {r2:.4f}")
            print("==============================================\n")
        else:
            a, b, r2, Y_pred = np.nan, np.nan, np.nan, None
            print(f"Not enough data points for linear regression ({label} trials).")

        return {
            "angles": unique_angles,
            "mean_rt": mean_rt,
            "acc": acc,
            "X": X,
            "Y_pred": Y_pred,
            "a": a,
            "b": b,
            "r2": r2,
            "label": label,
        }

    data_same = data[data[:, 1] == 1]
    data_diff = data[data[:, 1] == 0]

    stats_same = compute_stats_and_regression(data_same, label="same")
    stats_diff = compute_stats_and_regression(data_diff, label="different")

    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    # RT vs angle
    for stats, style in [(stats_same, "o-"), (stats_diff, "s-")]:
        angles = stats["angles"]
        mean_rt = stats["mean_rt"]
        label = stats["label"]

        axes[0].plot(angles, mean_rt, style, label=f"{label} (mean RT)")

        if stats["Y_pred"] is not None:
            X = stats["X"]
            Y_pred = stats["Y_pred"]
            a = stats["a"]
            r2 = stats["r2"]
            axes[0].plot(
                X,
                Y_pred,
                "--",
                label=f"{label} regression (slope={a:.1f} ms/°, R²={r2:.3f})",
            )

    axes[0].set_xlabel("Rotation angle (degrees)")
    axes[0].set_ylabel("Mean RT (ms)")
    axes[0].set_title("RT vs rotation angle\n(correct trials only)")
    axes[0].legend()

    # Accuracy vs angle
    for stats, style in [(stats_same, "o-"), (stats_diff, "s-")]:
        angles = stats["angles"]
        acc = stats["acc"]
        label = stats["label"]
        axes[1].plot(angles, acc, style, label=f"{label}")

    axes[1].set_xlabel("Rotation angle (degrees)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Accuracy vs rotation angle")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_filename, dpi=150)



def run_experiment(exp):
    expyriment.control.start(exp)

    kb = expyriment.io.Keyboard()

    instructions = expyriment.stimuli.TextScreen(
        "Instructions",
        "Two objects will appear on the screen.\n"
        "Your task is to decide whether they show the SAME object or\n"
        "DIFFERENT objects.\n\n"
        "Press F if they are DIFFERENT.\n"
        "Press J if they are THE SAME.\n\n"
        "Please respond as quickly and accurately as possible.\n\n"
        "Press SPACE to begin."
    )
    instructions.present()
    kb.wait_char(" ")

    fixation = expyriment.stimuli.FixCross(size=(20, 20), line_width=3)

    # Local copy of data for later analysis
    results = []

    for trial in exp.blocks[0].trials:

        fixation.present()
        exp.clock.wait(FIXATION_TIME)

        stim = trial.stimuli[0]
        stim.present()   # just show the composite stimulus

        exp.clock.reset_stopwatch()

        key, rt = kb.wait([expyriment.misc.constants.K_j,
                           expyriment.misc.constants.K_f])

        response_same = 1 if key == expyriment.misc.constants.K_j else 0
        correct = int(response_same == trial.get_factor("same"))

        angle = trial.get_factor("angle")
        same = trial.get_factor("same")
        rep = trial.get_factor("rep")

        row = [angle, same, rep, response_same, correct, rt]
        results.append(row)

        exp.data.add(row)

        exp.clock.wait(ISI_TIME)

    end = expyriment.stimuli.TextScreen("Finished", "Thank you!")
    end.present()
    exp.clock.wait(2000)

    expyriment.control.end()

    analyze_and_plot(results)


if __name__ == "__main__":
    exp = build_experiment()
    run_experiment(exp)