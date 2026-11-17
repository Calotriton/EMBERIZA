// Every option of bacpipe's config.yaml and settings.yaml, with a plain label and what it does (after the comments
// in bacpipe's own files). `key` is bacpipe's name. `off` marks an option that cannot have an effect in a browser:
// it is shown, so that nothing is hidden, and says why. tests/test_options.py checks that no bacpipe option is missing.
const NO_SERVER = 'bacpipe uses this for its own dashboard server. This page is the dashboard here.';
const NO_THREADS = 'bacpipe uses this to load audio on other processor threads. In a browser the audio is read between batches.';
const FIXED_PATH = 'A folder name inside the results. It is fixed here so that saved results are found again and bacpipe can open the ZIP.';

export const CONFIG_FIELDS = [
  { key: 'dim_reduction_model', label: 'Map the embeddings with', type: 'select',
    options: [{ value: 'umap', label: 'UMAP' }, { value: 't_sne', label: 't-SNE' }, { value: 'pca', label: 'PCA' }, { value: 'sparse_pca', label: 'Sparse PCA' }, { value: 'None', label: 'no map' }],
    help: 'How the embeddings are reduced to two or three dimensions for the map. UMAP and t-SNE keep neighbours together and are not linear. PCA is linear and gives the same map every time.' },
  { key: 'evaluation_task', label: 'Evaluation tasks', type: 'multi', options: ['clustering', 'probing'],
    help: 'Clustering groups the embeddings with k-means and scores the groups against every label. Probing trains a small classifier on the embeddings of annotated sounds and tests it. Probing needs an annotations file.' },
  { key: 'overwrite', label: 'Redo the evaluation', type: 'bool',
    help: 'Compute the labels, ground truth, clustering and probing again when they exist. Embeddings are never computed twice by this: see "Use saved embeddings".' },
];

export const SETTINGS_GROUPS = [
  { title: 'Embedding generation', fields: [
    { key: '__backend', label: 'Run the models on', type: 'select', options: [{ value: 'cpu', label: 'the processor' }, { value: 'gpu', label: 'the graphics card (experimental)' }],
      help: "The processor gives bacpipe's numbers to about five decimals and works everywhere. The graphics card (WebGPU) is faster where it works, and falls back on the processor where it does not. Its numbers can differ a little more. This stands for bacpipe's device setting." },
    { key: 'global_batch_size', label: 'Batch size', type: 'int', min: 1, max: 64,
      help: 'How many windows go through the model at once: 100,000 times this, divided by the samples in a window. Lower it if the tab runs out of memory.' },
    { key: 'padding', label: 'Fill a short last window by', type: 'select', options: ['wrap', 'constant', 'edge', 'reflect', 'symmetric', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum'],
      help: 'A recording rarely ends on a whole window. "wrap" repeats the recording from its start, "constant" adds silence. The names are numpy.pad\'s.' },
    { key: 'audio_suffixes', label: 'File endings read', type: 'list', help: 'Files with these endings count as recordings. Separate them with spaces.' },
    { key: 'check_if_already_processed', label: 'Use saved embeddings', type: 'bool',
      help: 'Recordings that were embedded before with the same model and dataset name are not embedded again, and a stopped run carries on. Switch off to embed everything again.' },
    { key: 'check_if_already_dim_reduced', label: 'Use a saved map', type: 'bool', help: 'Reuse the map made before with the same method. Switch off to make it again.' },
    { key: 'device', label: 'Device', type: 'text', off: 'bacpipe\'s choice of "cpu", "cuda" or "mps". Use "Run the models on" above.', help: 'The device bacpipe runs PyTorch and TensorFlow models on.' },
    { key: 'avoid_pipelined_gpu_inference', label: 'Load audio in step with the model', type: 'bool', off: NO_THREADS, help: 'Always work on one recording at a time.' },
    { key: 'nr_parallel_workers', label: 'Parallel workers', type: 'text', off: NO_THREADS, help: 'How many processor threads load audio.' },
    { key: 'rm_embedding_on_keyboard_interrupt', label: 'Delete embeddings when interrupted', type: 'bool', off: 'There is no keyboard interrupt here. Stop keeps what was embedded.', help: 'Remove a short unfinished run when bacpipe is interrupted from the keyboard.' },
  ] },
  { title: 'Your annotations', note: 'Put annotations.csv in the top folder of the recordings, with the columns audiofilename, start, end and label:species. More label: columns are allowed.', fields: [
    { key: 'annotations_filename', label: 'Annotations file', type: 'text', help: 'The name of the table of annotations in the top folder of the recordings.' },
    { key: 'label_column', label: 'Main label', type: 'text', help: 'Which label: column clustering and probing are scored against. "species" means the column label:species.' },
    { key: 'only_embed_annotations', label: 'Embed only annotated sounds', type: 'bool', help: 'Make one embedding per annotation instead of windowing the whole recording. A short annotation is filled to a full window, a long one gives several embeddings.' },
    { key: 'min_annotation_length', label: 'Shortest annotation used (s)', type: 'float', min: 0, help: 'Annotations shorter than this are left out of the ground truth.' },
  ] },
  { title: 'Labels from file names and folders', fields: [
    { key: 'metadata_label_keys', label: 'Labels to make', type: 'multi', options: ['time_of_day', 'week_of_year', 'day_of_year', 'continuous_timestamp', 'parent_directory', 'audio_file_name'],
      help: 'Labels bacpipe works out for every embedding, to colour the map with. Dates and times are read from the digits in the file name, for example 20250503_031300.' },
  ] },
  { title: "The model's own classifier", fields: [
    { key: 'run_pretrained_classifier', label: 'Run the built-in classifier', type: 'bool', help: 'Both models come with a species classifier. Its confidence scores are saved with the embeddings and give the map a "default classifier" colouring.' },
    { key: 'classifier_threshold', label: 'Keep confidence scores above', type: 'float', min: 0, max: 1, step: 0.05, help: 'Only scores above this are saved. A confidence score is not a probability, and what a value means differs between species.' },
    { key: 'max_labels_per_timestamp', label: 'Most species kept per window', type: 'int', min: 1, max: 1000, help: 'At most this many species are saved for one window, highest scores first.' },
    { key: 'save_raven_tables', label: 'Write Raven selection tables', type: 'bool', help: 'One table of predictions per recording, to open over the spectrogram in Raven.' },
    { key: 'min_label_occurrences', label: 'Fewest sounds per label', type: 'int', min: 1, help: 'With "Filter rare labels" on, labels with fewer annotated sounds than this are treated as noise.' },
    { key: 'bool_filter_labels', label: 'Filter rare labels', type: 'bool', help: 'Leave labels with too few annotated sounds out of the ground truth.' },
  ] },
  { title: 'Probing', fields: [
    { key: 'train_ratio', label: 'Share for training', type: 'float', min: 0.05, max: 0.9, step: 0.05, help: 'The share of each label\'s annotated windows used to train the probe. What is left after training and testing is the validation share.' },
    { key: 'test_ratio', label: 'Share for testing', type: 'float', min: 0.05, max: 0.9, step: 0.05, help: 'The share of each label\'s annotated windows the probe is scored on.' },
    { heading: 'Linear probe' },
    { key: 'probe_configs.config_1.bool', label: 'Train it', type: 'bool', help: 'One linear layer on top of the embeddings.' },
    { key: 'probe_configs.config_1.learning_rate', label: 'Learning rate', type: 'float', min: 0.000001, max: 1, help: 'The step size of the Adam optimiser.' },
    { key: 'probe_configs.config_1.batch_size', label: 'Batch size', type: 'int', min: 1, help: 'Windows per training step.' },
    { key: 'probe_configs.config_1.num_epochs', label: 'Epochs', type: 'int', min: 1, max: 1000, help: 'Passes over the training windows.' },
    { key: 'probe_configs.config_1.shuffle', label: 'Shuffle each epoch', type: 'bool', help: 'Present the training windows in a new random order each epoch.' },
    { heading: 'k nearest neighbours probe' },
    { key: 'probe_configs.config_2.bool', label: 'Fit it', type: 'bool', help: 'Each test window gets the commonest label among its nearest training windows.' },
    { key: 'probe_configs.config_2.n_neighbors', label: 'Neighbours', type: 'int', min: 1, max: 200, help: 'How many nearest training windows vote.' },
    { key: 'save_probe', label: 'Save the trained probe', type: 'bool', help: 'Write linear_probe.pt in PyTorch\'s file format, with the list of classes, so that bacpipe or PyTorch can use it.' },
  ] },
  { title: 'Clustering', fields: [
    { key: 'evaluate_with_silhouette', label: 'Also the silhouette score', type: 'bool', help: 'A score of how separate the clusters are. Slow on large datasets.' },
    { heading: 'k-means' },
    { key: 'clust_configs.config_1.bool', label: 'Run it', type: 'bool', help: 'Group the embeddings into k clusters.' },
    { key: 'clust_configs.config_1.params.n_clusters', label: 'Clusters', type: 'text', parse: (v) => (/^\d+$/.test(v.trim()) ? Number(v) : 'None'),
      help: 'How many clusters. "None" takes the number of different labels in the ground truth.' },
    { heading: 'HDBSCAN' },
    { key: 'clust_configs.config_2.bool', label: 'Run it', type: 'bool', off: 'bacpipe 1.3.5 has this option but does not run HDBSCAN yet.', help: 'Density-based clustering.' },
    { key: 'clust_configs.config_2.params.min_cluster_size', label: 'Smallest cluster', type: 'int', off: 'bacpipe 1.3.5 does not run HDBSCAN yet.', help: 'The fewest points that make a cluster.' },
    { key: 'clust_configs.config_2.params.min_samples', label: 'Minimum samples', type: 'int', off: 'bacpipe 1.3.5 does not run HDBSCAN yet.', help: 'How conservative the clustering is.' },
    { key: 'clust_configs.config_2.params.metric', label: 'Metric', type: 'text', off: 'bacpipe 1.3.5 does not run HDBSCAN yet.', help: 'The distance between embeddings.' },
  ] },
  { title: 'Maps and plots', fields: [
    { key: 'visualization_dimensions', label: 'Dimensions of the map', type: 'select', options: [{ value: 2, label: '2' }, { value: 3, label: '3' }], help: 'bacpipe makes UMAP maps in two or three dimensions. Its other methods always give two.' },
    { key: 'nr_predictions_to_display', label: 'Predictions shown per window', type: 'int', min: 0, max: 20, help: 'How many of the classifier\'s predictions are listed for the window you click.' },
    { key: 'max_nr_categories', label: 'Most labels with their own colour', type: 'int', min: 2, max: 200, help: 'With more different labels than this, the map uses a colour scale instead of one colour per label.' },
    { key: 'color_continuous', label: 'Colour scale of the map', type: 'select', options: ['Twilight', 'Viridis', 'Cividis', 'Portland', 'Electric', 'Hot', 'Jet', 'Rainbow'], help: 'Used when there are too many labels for single colours. Twilight is circular, which suits the time of day.' },
    { key: 'spec_colorscale', label: 'Colours of the spectrogram', type: 'select', options: ['Viridis', 'Cividis', 'Greys', 'Hot', 'Electric', 'Blackbody', 'Portland', 'Jet'], help: 'The colour scale of the spectrogram of the window you click.' },
    { key: 'embed_fig_height', label: 'Height of the map (px)', type: 'int', min: 300, max: 1400, help: 'How tall the map is drawn.' },
    { key: 'spectrogram_plot_height', label: 'Height of the spectrogram (px)', type: 'int', min: 300, max: 1200, help: 'How tall the spectrogram is drawn.' },
    { key: 'heatmap_fig_height', label: 'Height of the heatmap (px)', type: 'int', min: 300, max: 1400, off: 'The heatmap grows with the number of species here.', help: 'How tall bacpipe draws the predictions heatmap.' },
    { key: 'accordion_width', label: 'Panel width (px)', type: 'int', off: NO_SERVER, help: 'The width of the panels of bacpipe\'s dashboard.' },
    { key: 'dashboard_port', label: 'Dashboard port', type: 'int', off: NO_SERVER, help: 'The port of bacpipe\'s dashboard server.' },
    { key: 'dashboard_address', label: 'Dashboard address', type: 'text', off: NO_SERVER, help: 'The address of bacpipe\'s dashboard server.' },
    { key: 'dashboard_websocket_origin', label: 'Embed the dashboard', type: 'text', off: NO_SERVER, help: 'Lets bacpipe\'s dashboard be embedded in another website.' },
  ] },
  { title: 'Change of speed, for ultrasound', note: 'Still in testing in bacpipe.', fields: [
    { key: 'bool_change_speed', label: 'Slow down or speed up the sound', type: 'bool', help: 'Play the recordings at another speed before the model hears them, so that a bird model can be tried on ultrasound.' },
    { key: 'new_speed', label: 'New speed', type: 'float', min: 0.001, max: 100, help: '0.1 plays ten times slower, which lowers every frequency ten times.' },
  ] },
  { title: 'Folders', fields: [
    { key: 'main_results_dir', label: 'Results folder', type: 'text', off: 'Results are kept in this browser and downloaded as a ZIP named bacpipe_results.', help: 'Where bacpipe writes its results.' },
    { key: 'embed_parent_dir', label: 'Embeddings folder', type: 'text', off: FIXED_PATH, help: 'The folder of the embeddings inside the results.' },
    { key: 'dim_reduc_parent_dir', label: 'Maps folder', type: 'text', off: FIXED_PATH, help: 'The folder of the reduced embeddings inside the results.' },
    { key: 'evaluations_dir', label: 'Evaluations folder', type: 'text', off: FIXED_PATH, help: 'The folder of labels, predictions, clustering and probing inside the results.' },
    { key: 'model_base_path', label: 'Model checkpoints folder', type: 'text', off: 'The page downloads the model files and keeps them in the browser.', help: 'Where bacpipe keeps model files.' },
    { key: 'testing', label: 'Test mode', type: 'bool', off: "bacpipe's own test mode: it uses fixed test files and deletes the results.", help: 'Used by bacpipe\'s tests.' },
  ] },
];

// config.yaml keys that are not form fields, and where they are in the page
export const CONFIG_ELSEWHERE = { audio_dir: 'step 1, Recordings', models: 'step 2, Models', dashboard: 'the page is the dashboard', already_computed: 'Saved in this browser, Open' };

export const HELP = {
  clustering: "These scores compare the k-means clusters of the embeddings with a label, so they need something to compare with. Without annotations, bacpipe uses the labels from file names and folders, such as the time of day. That shows whether the embeddings are organised by that label.\n\nAn adjusted mutual information near 1 means the clusters and the label share most of their information. Near 0 means the embeddings are not organised that way. Treat it as a quick way to explore, not as a result. (After bacpipe's own explanation.)",
  probing: "Probing needs annotations: a file annotations.csv in the top folder of the recordings, with the columns audiofilename, start, end and label:species. The annotated windows are split into training, validation and test parts. A linear probe and a k nearest neighbours probe are fitted on the embeddings of the training part, and scored per class on the test part. (After bacpipe's own explanation.)",
};
