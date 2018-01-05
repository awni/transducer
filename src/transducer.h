
   
/*
 * The activations should be shape (time, minibatch, alphabet)
 * in C order.
 * 
 * *NB* We assume the blank_label is the last index in the alphabet.
 */
void transduce(THFloatTensor *th_log_probs,
               THIntTensor *th_labels,
               THIntTensor *th_input_lengths,
               THIntTensor *th_label_lengths,
               THFloatTensor *th_costs,
               THFloatTensor *th_grads,
               int blank);

