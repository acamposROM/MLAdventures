#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Transformer model

typedef struct {
  int dim; // transformer dimension 
  int hidden_dim; // for ffn layers (feed foward network)
  int n_layers; // number of layers
  int n_heads; // number of query heads (is this a transformer thing?)
  int n_kv_heads; // number of key/value heads (can be < query heads because of the mulityquery)
}
void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  
  // default parameters
  char * checkpoint_path = NULL; // e.g out/model.bin
  char * tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;
  float topp = 0.9f;
  int steps = 256;
  char *prompt = NULL;
  unsigned long long rng_seed = 0;
  char *mode = "generate";
  char *system_prompt = NULL;
  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }

  for (int i = 2; i < argc; i+=2) {
    // do some basic validation
    if (i + 1 >= argc) { error_usage(); } // must have arg after flag
    if (argv[i][0] != '-') { error_usage(); } // must start with dash
    if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
    
    // read in the args
    if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
    else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
    else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
    else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
    else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
    else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
    else { error_usage(); }
  }
  // parameter validation/overrides
  if (rng_seed <= 0 ) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  //build the Transformer via the model .bin file
  
  Transformer transformer;
  return 0;
}
