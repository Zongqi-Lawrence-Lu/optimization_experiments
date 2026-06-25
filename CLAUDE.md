1. This is a project that lets me test different optimization methods on centralized/distributed settings.
This will allow me to compare optimization methods empricially in different settings and inform theoretical
results. The runs will be done remotely on a GPU cluster. For basic runs I will likely allocate a single
A40 GPU, but we could upgrade to multiple GPU or A100 (80gb) if necessary. The maximum wall time allowed by
the cluster policy is 12 hours.

2. There are four main settings of the experiments I would like to run on. Make sure the modes are available
and the pipeline is ready for any of these.
a. Convex setting with linear regression. Use a synthetic linear regression task where the feature matrix has
Gaussian entries and labels are generated from a true weight vector plus injected noise. There are two further
noise regimes: a light-tailed Gaussian and a heavy-tailed Student-t distribution, and also vary noise scale.
b. Convex settings with tokens. Mimic token frequency distributions in natural language by splitting features
into a small "common" group with high activation probability and a large "rare" group with low activation
probability.
c. Non-convex language model. Finetune RoBERTa on the full GLUE benchmark, covering tasks like sentiment
analysis, textual entailment, sentence similarity, and natural language inference. 
d. Generative languae model. Finetune T5 on WMT machine translation data (TED Talks and News Commentary, English
to German and French)

3. There are many conditions I would like to run on. This means that I will select one condition each run and
compare them.
a. I need to be able to run on centralied or distributed settings. Leave the number of nods and synchronization
time stamp open for me to decide. Also note that the outer node optimization method should be selectable.
b. There are two main types of optimizations I will use. The first is clipping. The clipping can further be sub-
divided into a two types. i) Upper clipping, where we only set a upper threshold and clip if it exceeds the
threshold ii) Biclip, where we give both lower clip and upper clip, so when the norm is too small, we will 
increase it to the lower level. The threshold could be dynamic or fixed, and there should be a way to easily
configure the settings. Keep in mind that in distributed settings, both inner and outer node could use clipping.
By default, the threshold should be fixed.
c. The second is adaptive methods. Again, this might be invoked by both inner and outer node in distributed
settings. I want to be able to select the parmeters. The methods should include adagrad, adagrad-norm, rmsprop,
adam, and adamW. Other than the two types, no operation and using vanilla SGD should also be allowed.
d. When using clipping and adaptive methods, I need to do it in three modes: coordinate-wise, layer-wise, or 
globally. For instance, a layer-wise upper clipping means that we may set separate clipping threshold for each
layer and clip the gradient for each layer individually. Globally means that the entire gradient will be treated
as a whole.
e. A robust hyperparmeter sweep must be enabled. Hyperparameters are tuned via a two-stage grid search — a coarse
sweep first to locate a good region, then a finer sweep around it. The clipping thresholds are swept alongside
learning rates, and adaptive optimizer epsilon values are also included in the grid. The best configuration per 
method per dataset is then used for the reported numbers.
f. There should also be an optional learning rate warmup mechanism. If enabled, the maximum learning rate should
be configurable, so is the duration of the warmup period. By default, the warmup should be linear.

4. There needs to be specialized folder for raw input data, output (such as model weights), results (which are more
organized than output, such as json), and for plots. Write a README.md to document the file structure. There should
also be strong plotting infrastructure available. For instance, I would like the heatmap for the hyperparmeter sweep
and also for the model run, which should both record (in results/ folder) and plot (in plots/ folder) the trend for
loss, train/val/test accuracy. You are free to create additional folder structures.

5. As a general guideline, write clean and modular code. There should be strong logging and restart mechanism if the 
duration of the run is expected over an hour. As general rule, a checkpoint should be about every 10 minutes.