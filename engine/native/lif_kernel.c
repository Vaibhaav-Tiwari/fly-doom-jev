/* LIF kernel for the full MaleCNS graph (211,577 neurons / 26.0M edges).
 *
 * Written from scratch for jev-doom-fly; the compiled-kernel architecture
 * (clang at setup, CSR passed by pointer) follows the approach used by
 * DOOMFLY (MIT) — see PROVENANCE.md.
 *
 * Dynamics (all neurons, every step; dt in ms):
 *   g        *= exp(-dt/tau_syn)                       synaptic drive decay
 *   v         = rest + (v-rest)*exp(-dt/tau_v) + (drive + g*gsyn) * (1-av)
 *   spike if  v >= threshold and not refractory
 *   on spike: enqueue for delivery after `delay_steps`, v = reset,
 *             refractory = ref_steps
 * Delivery: for each presynaptic spike, g[post] += weight over outgoing CSR.
 */
#include <stdint.h>

typedef struct {
    int64_t n;
    const int64_t *ptr;      /* (n+1) CSR outgoing edges by presynaptic neuron */
    const int32_t *post;     /* (E) */
    const float *weight;     /* (E) signed */
    float *v;                /* (n) membrane potential, mV-ish */
    float *g;                /* (n) synaptic drive accumulator */
    int16_t *refractory;     /* (n) steps remaining */
    const float *drive;      /* (n) external input current */
    int32_t *counts;         /* (n) spikes emitted during this call */
    int32_t *queue;          /* (slots * cap) ring buffer of presynaptic indices */
    int32_t *queue_count;    /* (slots) */
    int64_t cursor;
    int32_t slots;           /* ring size = delay_steps + 1 */
    int32_t delay_steps;
    int32_t ref_steps;
    float rest, threshold, reset, av, ag, gsyn;
} LifState;

void lif_advance(LifState *s, int64_t steps) {
    const int64_t n = s->n;
    for (int64_t step = 0; step < steps; step++) {
        int32_t slot = (int32_t)(s->cursor % s->slots);
        /* 1. deliver spikes scheduled for this step */
        int64_t row = (int64_t)slot * s->n;  /* queue row: slot * n entries */
        for (int32_t q = 0; q < s->queue_count[slot]; q++) {
            int32_t prei = s->queue[row + q];
            for (int64_t e = s->ptr[prei]; e < s->ptr[prei + 1]; e++)
                s->g[s->post[e]] += s->weight[e];
        }
        s->queue_count[slot] = 0;

        /* 2. integrate + threshold */
        int32_t future = (int32_t)((s->cursor + s->delay_steps) % s->slots);
        int64_t frow = (int64_t)future * s->n;
        for (int64_t i = 0; i < n; i++) {
            if (s->refractory[i] > 0) { s->refractory[i]--; continue; }
            float g = s->g[i] * s->ag;
            float drive = s->drive[i] + g * s->gsyn;
            float v = s->rest + (s->v[i] - s->rest) * s->av + drive * (1.0f - s->av);
            s->g[i] = g;
            if (v >= s->threshold) {
                v = s->reset;
                s->refractory[i] = (int16_t)s->ref_steps;
                s->counts[i]++;
                int32_t c = s->queue_count[future];
                if (c < (int32_t)s->n) {  /* ring row cap: n spikes/step max */
                    s->queue[frow + c] = (int32_t)i;
                    s->queue_count[future] = c + 1;
                }
            }
            s->v[i] = v;
        }
        s->cursor++;
    }
}
