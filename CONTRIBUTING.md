# Contributing

Contributions are welcome — a new analysis, a clearer explanation, a bug you
found, a format you needed to read.

## What you keep

**Your data is yours.** Nothing in this repository sends anything anywhere.
The scripts read files on your machine and write figures next to them. We have
no interest in what you record, what you analyse, or what you find.

**What you build is yours.** Use these scripts in your thesis, your startup,
your course, your product. Apache-2.0 permits commercial use, and we mean it
to. Build something and sell it.

The only thing the licence asks is that if you distribute software derived from
this, you carry the `NOTICE` file's attribution with it — in the source, the
documentation, or somewhere the work itself displays. That is the whole
obligation. You are not required to open your own code, share your
improvements, or tell us anything.

If you publish work that used these scripts, a citation is appreciated (see
[CITATION.cff](CITATION.cff)) but is a courtesy, not a requirement.

## What we are trying to keep true

This repository exists so that someone can **read** an analysis and understand
it. [MNE-Python](https://mne.tools) already does everything here, better and
with more validation behind it. The only thing these scripts offer that MNE
does not is that you can follow them start to finish in an afternoon.

That shapes what fits:

**Dependencies stop at NumPy, SciPy and Matplotlib.** A reader you have to
install a toolkit to use is a worse interface to the toolkit. If an addition
needs a fourth dependency, it probably belongs in a project built on top of
this one rather than in it.

**Comments explain the signal processing, not the syntax.** `# loop over
channels` is noise. "A longer window buys finer frequency resolution but leaves
fewer segments to average" is the point. If a choice could reasonably have gone
another way — a window length, a band edge, a threshold — say why it went the
way it did.

**No hardware assumptions.** No fixed gain, no assumed sampling rate, no
hard-coded electrode names. Channel names are typed by whoever recorded the
data, so they are always arguments. A script should work on a recording from
equipment we have never seen.

**Fail loudly, never silently.** A sampling rate that gets guessed rescales
every frequency in the result and nothing looks wrong. If a value cannot be
determined, raise, and say in the message what to pass instead.

## Adding an analysis

The shape to follow, from any of the existing scripts:

1. A module docstring that explains **what the measurement is and why anyone
   cares**, before any code. Include usage examples and references to the
   literature.
2. An `analyse()` function that takes a `Recording` and returns a dict. Every
   key carries its unit in the name — `alpha_power_closed_uv2`, not
   `alpha_power` — so a number cannot be mistaken for a different quantity.
3. A `report()` that prints the result *and* how to read it. If a value falls
   in a range that means something, say so. If it is backwards, say that too,
   and say what usually causes it.
4. A `plot()`, if a figure helps.
5. A generator in `scripts/synth.py` producing data with a **known answer**,
   and tests asserting the analysis recovers it.

That last one is not optional, and it is the most useful part. Tests here check
results against answers known in closed form, not against whatever the code
printed last time. A sinusoid of peak amplitude *A* must return band power
*A*²/2. Aligned phases must give a phase-locking value near 1, and scattered
ones near chance. If you cannot state what the right answer is, the analysis is
not ready.

## Practicalities

```bash
pip install numpy scipy matplotlib pytest
python -m pytest tests/
```

- Branch from `main`, open a pull request.
- Keep commits focused; a commit message should say *why*, since the diff
  already says what.
- CI runs on Ubuntu deliberately, because its filesystem is case-sensitive and
  catches a class of import bug that never appears on macOS.
- Nothing needs signing. There is no contributor licence agreement: your
  contribution is under Apache-2.0 like the rest of the repository, and you keep
  your copyright in it.

## Sharing what you build

We are planning development sprints and hackathons around biosignal analysis.
If you build something on these scripts, we would like to hear about it — not
because we want a share, but because a body of worked examples is worth more
than a body of code.

The practical form that takes is not settled yet. For now: open an issue, or
open a pull request adding your analysis if it generalises beyond your own
recording.

## Reporting a problem

Open an issue with the command you ran, what you expected, and what happened.
If it involves a recording, the **format** matters more than the data — the
first line of the file, the channel names, the sampling rate. Please do not
attach recordings of identifiable people.

A wrong number is a more serious bug than a crash, so if a result looks
implausible rather than erroring, that is worth reporting too.
