"""
CUDA kernel definitions for YS-vision-tools GPU acceleration

Custom CUDA kernels for high-performance operations on RTX 5090.
Compiled on-demand using CuPy's RawKernel interface.

Author: Yambo Studio
"""

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# ==============================================================================
# Echo Layer Kernels - Temporal Accumulation with EMA
# ==============================================================================

ECHO_EMA_UPDATE_KERNEL = r'''
extern "C" __global__
void echo_ema_update(
    const float* input_rgba,    // Input layer RGBA premultiplied (H*W*4)
    const float* state_rgba,    // Previous state RGBA premultiplied (H*W*4)
    const unsigned char* state_age,  // Previous age buffer (H*W)
    float* output_rgba,         // Output layer RGBA premultiplied (H*W*4)
    float* new_state_rgba,      // Updated state RGBA premultiplied (H*W*4)
    unsigned char* new_state_age,    // Updated age buffer
    const float decay,
    const float boost_on_new,
    const float exposure,
    const int max_age,
    const int width,
    const int height,
    const int clamp_hard        // 1=hard clamp, 0=soft clip
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx_rgba = idx * 4;

    // Read input RGBA (premultiplied)
    float in_r = input_rgba[idx_rgba + 0];
    float in_g = input_rgba[idx_rgba + 1];
    float in_b = input_rgba[idx_rgba + 2];
    float in_a = input_rgba[idx_rgba + 3];

    // Read previous state RGBA (premultiplied)
    float prev_r = state_rgba[idx_rgba + 0];
    float prev_g = state_rgba[idx_rgba + 1];
    float prev_b = state_rgba[idx_rgba + 2];
    float prev_a = state_rgba[idx_rgba + 3];
    unsigned char prev_age = state_age[idx];

    // Apply exposure boost to INPUT first (makes fresh trails brighter)
    float input_boost = (in_a > 0.001f) ? (1.0f + boost_on_new) * exposure : 1.0f;
    in_r *= input_boost;
    in_g *= input_boost;
    in_b *= input_boost;
    in_a *= input_boost;

    // EMA update (directly on premultiplied RGBA)
    // This preserves color information correctly
    float new_r = decay * prev_r + (1.0f - decay) * in_r;
    float new_g = decay * prev_g + (1.0f - decay) * in_g;
    float new_b = decay * prev_b + (1.0f - decay) * in_b;
    float new_a = decay * prev_a + (1.0f - decay) * in_a;

    // Age tracking (reset on new input)
    unsigned char new_age = (in_a > 0.001f) ? 0 : min(prev_age + 1, (unsigned char)max_age);

    // Age-based fade (optional: fade out old trails)
    float age_fade = (new_age < max_age) ? 1.0f : fmaxf(0.0f, 1.0f - (float)(new_age - max_age) / 10.0f);
    new_r *= age_fade;
    new_g *= age_fade;
    new_b *= age_fade;
    new_a *= age_fade;

    // Apply clamping to output
    float out_r = new_r;
    float out_g = new_g;
    float out_b = new_b;
    float out_a = new_a;

    if (clamp_hard) {
        // Hard clamp [0, 1]
        out_r = fminf(fmaxf(out_r, 0.0f), 1.0f);
        out_g = fminf(fmaxf(out_g, 0.0f), 1.0f);
        out_b = fminf(fmaxf(out_b, 0.0f), 1.0f);
        out_a = fminf(fmaxf(out_a, 0.0f), 1.0f);
    } else {
        // Soft clip RGB (tanh-like), hard clamp alpha
        auto softclip = [](float x) { 
            return (x > 0.0f) ? (x / (1.0f + x)) : (x / (1.0f - x));
        };
        out_r = softclip(out_r);
        out_g = softclip(out_g);
        out_b = softclip(out_b);
        out_a = fminf(fmaxf(out_a, 0.0f), 1.0f);
    }

    // Write output (premultiplied RGBA)
    output_rgba[idx_rgba + 0] = out_r;
    output_rgba[idx_rgba + 1] = out_g;
    output_rgba[idx_rgba + 2] = out_b;
    output_rgba[idx_rgba + 3] = out_a;

    // Write updated state (premultiplied RGBA)
    new_state_rgba[idx_rgba + 0] = out_r;
    new_state_rgba[idx_rgba + 1] = out_g;
    new_state_rgba[idx_rgba + 2] = out_b;
    new_state_rgba[idx_rgba + 3] = out_a;
    new_state_age[idx] = new_age;
}
'''


# ==============================================================================
# SDF Text Rendering Kernel
# ==============================================================================

SDF_TEXT_RENDER_KERNEL = r'''
__device__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

extern "C" __global__
void sdf_text_render(
    const float* atlas_sdf,     // SDF atlas texture (atlas_size*atlas_size)
    const float* glyph_quads,   // Glyph quad data: [x, y, u, v, w, h, r, g, b, a] * N
    const int num_glyphs,
    float* output_rgba,         // Output RGBA layer (H*W*4)
    const int width,
    const int height,
    const int atlas_size,
    const float sdf_threshold,  // Distance threshold for edge
    const float stroke_width    // Stroke width in pixels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;
    float out_r = 0.0f, out_g = 0.0f, out_b = 0.0f, out_a = 0.0f;

    // Check each glyph quad
    for (int i = 0; i < num_glyphs; i++) {
        const float* quad = glyph_quads + i * 10;
        float gx = quad[0], gy = quad[1];  // Position
        float u = quad[2], v = quad[3];    // Atlas UV top-left
        float gw = quad[4], gh = quad[5];  // Glyph size
        float r = quad[6], g = quad[7], b = quad[8], a = quad[9];  // Color

        // Check if pixel is inside glyph bounds
        if (x >= gx && x < gx + gw && y >= gy && y < gy + gh) {
            // Compute UV coordinates
            float local_u = u + (x - gx) / gw;
            float local_v = v + (y - gy) / gh;

            // Sample SDF atlas (bilinear interpolation)
            int atlas_x = (int)(local_u * atlas_size);
            int atlas_y = (int)(local_v * atlas_size);
            atlas_x = max(0, min(atlas_x, atlas_size - 1));
            atlas_y = max(0, min(atlas_y, atlas_size - 1));

            float sdf_value = atlas_sdf[atlas_y * atlas_size + atlas_x];

            // Compute alpha from SDF
            float edge_alpha = smoothstep(sdf_threshold - 0.02f, sdf_threshold + 0.02f, sdf_value);

            // Optional stroke
            float stroke_alpha = 0.0f;
            if (stroke_width > 0.0f) {
                stroke_alpha = smoothstep(sdf_threshold - stroke_width - 0.02f,
                                         sdf_threshold - stroke_width + 0.02f,
                                         sdf_value);
            }

            float final_alpha = max(edge_alpha, stroke_alpha) * a;

            // Alpha blend with existing output
            if (final_alpha > 0.01f) {
                float src_a = final_alpha;
                float dst_a = out_a;
                float blend_a = src_a + dst_a * (1.0f - src_a);

                if (blend_a > 0.0f) {
                    out_r = (r * src_a + out_r * dst_a * (1.0f - src_a)) / blend_a;
                    out_g = (g * src_a + out_g * dst_a * (1.0f - src_a)) / blend_a;
                    out_b = (b * src_a + out_b * dst_a * (1.0f - src_a)) / blend_a;
                    out_a = blend_a;
                }
            }
        }
    }

    // Write premultiplied output
    output_rgba[idx + 0] = out_r * out_a;
    output_rgba[idx + 1] = out_g * out_a;
    output_rgba[idx + 2] = out_b * out_a;
    output_rgba[idx + 3] = out_a;
}
'''


# ==============================================================================
# Pixel Sorting Kernels
# ==============================================================================

COMPUTE_METRIC_KERNEL = r'''
extern "C" __global__
void compute_metric(
    const float* image_rgb,     // Input RGB image (H*W*3)
    float* metric_out,          // Output metric (H*W)
    const int width,
    const int height,
    const int mode              // 0=luma, 1=saturation, 2=hue, 3=sobel
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx_rgb = idx * 3;

    float r = image_rgb[idx_rgb + 0];
    float g = image_rgb[idx_rgb + 1];
    float b = image_rgb[idx_rgb + 2];

    float metric = 0.0f;

    if (mode == 0) {
        // Luma (Rec. 709)
        metric = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    } else if (mode == 1) {
        // Saturation (HSV)
        float cmax = fmaxf(r, fmaxf(g, b));
        float cmin = fminf(r, fminf(g, b));
        metric = (cmax > 0.0f) ? (cmax - cmin) / cmax : 0.0f;
    } else if (mode == 2) {
        // Hue (HSV)
        float cmax = fmaxf(r, fmaxf(g, b));
        float cmin = fminf(r, fminf(g, b));
        float delta = cmax - cmin;

        if (delta > 0.0f) {
            if (cmax == r) {
                metric = fmodf((g - b) / delta + 6.0f, 6.0f) / 6.0f;
            } else if (cmax == g) {
                metric = ((b - r) / delta + 2.0f) / 6.0f;
            } else {
                metric = ((r - g) / delta + 4.0f) / 6.0f;
            }
        }
    } else if (mode == 3) {
        // Sobel magnitude (approximate)
        // Simple gradient approximation using neighbors
        float gx = 0.0f, gy = 0.0f;

        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            int idx_left = idx_rgb - 3;
            int idx_right = idx_rgb + 3;
            int idx_up = idx_rgb - width * 3;
            int idx_down = idx_rgb + width * 3;

            float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            float luma_left = 0.2126f * image_rgb[idx_left] + 0.7152f * image_rgb[idx_left+1] + 0.0722f * image_rgb[idx_left+2];
            float luma_right = 0.2126f * image_rgb[idx_right] + 0.7152f * image_rgb[idx_right+1] + 0.0722f * image_rgb[idx_right+2];
            float luma_up = 0.2126f * image_rgb[idx_up] + 0.7152f * image_rgb[idx_up+1] + 0.0722f * image_rgb[idx_up+2];
            float luma_down = 0.2126f * image_rgb[idx_down] + 0.7152f * image_rgb[idx_down+1] + 0.0722f * image_rgb[idx_down+2];

            gx = luma_right - luma_left;
            gy = luma_down - luma_up;
        }

        metric = sqrtf(gx * gx + gy * gy);
    }

    metric_out[idx] = metric;
}
'''


CREATE_TRACK_MASK_KERNEL = r'''
extern "C" __global__
void create_track_mask(
    const float* tracks,        // Track positions (N*2)
    const int num_tracks,
    unsigned char* mask_out,    // Output binary mask (H*W)
    const int width,
    const int height,
    const float radius,         // Region radius
    const int use_box,          // 0=circle, 1=box
    const float box_w,
    const float box_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned char is_inside = 0;

    // Check against all tracks
    for (int i = 0; i < num_tracks; i++) {
        float tx = tracks[i * 2 + 0];
        float ty = tracks[i * 2 + 1];

        if (use_box) {
            // Box region
            if (fabsf(x - tx) <= box_w * 0.5f && fabsf(y - ty) <= box_h * 0.5f) {
                is_inside = 1;
                break;
            }
        } else {
            // Circular region
            float dx = x - tx;
            float dy = y - ty;
            if (dx * dx + dy * dy <= radius * radius) {
                is_inside = 1;
                break;
            }
        }
    }

    mask_out[idx] = is_inside ? 255 : 0;
}
'''


# ==============================================================================
# Kernel Compilation Cache
# ==============================================================================

_compiled_kernels = {}


def get_compiled_kernel(kernel_name: str, kernel_code: str):
    """
    Get compiled CUDA kernel (cached)

    Args:
        kernel_name: Kernel function name
        kernel_code: CUDA kernel source code

    Returns:
        Compiled CuPy RawKernel
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot compile CUDA kernels")

    cache_key = kernel_name

    if cache_key not in _compiled_kernels:
        _compiled_kernels[cache_key] = cp.RawKernel(kernel_code, kernel_name)

    return _compiled_kernels[cache_key]


def clear_kernel_cache():
    """Clear compiled kernel cache"""
    global _compiled_kernels
    _compiled_kernels = {}