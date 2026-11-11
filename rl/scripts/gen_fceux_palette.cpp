// Dump FCEUX's default NTSC colour table with emphasis (512 x RGB): palette[] from palettes/palettes.h and
// ApplyDeemphasisBisqwit copied verbatim from FCEUX src/palette.cpp (FCEUX ea6ed69b, GPL-2.0). This is what
// ChoosePalette() uses when no user/NTSC/game palette is set (the fresh-profile default).
// build: c++ -O0 -I.cache/fceux-compat-build-v2/src rl/scripts/gen_fceux_palette.cpp -o .cache/rl/gen_fceux_palette
#include <cmath>
#include <cstdint>
#include <cstdio>
typedef std::uint8_t uint8;
typedef std::uint8_t u8;
#include "palette.h"
#include "palettes/palettes.h"
float bisqwit_gammafix(float f, float gamma) { return f < 0.f ? 0.f : std::pow(f, 2.2f / gamma); }
int bisqwit_clamp(int v) { return v<0 ? 0 : v>255 ? 255 : v; }

// Calculate the luma and chroma by emulating the relevant circuits:
int bisqwit_wave(int p, int color) { return (color+p+8)%12 < 6; }

static void ApplyDeemphasisBisqwit(int entry, u8& r, u8& g, u8& b)
{
	if(entry<64) return;
	int myr=0, myg=0, myb=0;
	// The input value is a NES color index (with de-emphasis bits).
	// We need RGB values. Convert the index into RGB.
	// For most part, this process is described at:
	//    http://wiki.nesdev.com/w/index.php/NTSC_video

	// Decode the color index
	int color = (entry & 0x0F), level = color<0xE ? (entry>>4) & 3 : 1;

	// Voltage levels, relative to synch voltage
	static const float black=.518f, white=1.962f, attenuation=.746f,
		levels[8] = {.350f, .518f, .962f,1.550f,  // Signal low
		1.094f,1.506f,1.962f,1.962f}; // Signal high

	float lo_and_hi[2] = { levels[level + 4 * (color == 0x0)],
		levels[level + 4 * (color <  0xD)] };



	//fceux alteration: two passes
	//1st pass calculates bisqwit's base color
	//2nd pass calculates it with deemph
	//finally, we'll do something dumb: find a 'scale factor' between them and apply it to the input palette. (later, we could pregenerate the scale factors)
	//whatever, it gets the job done.
	for(int pass=0;pass<2;pass++)
	{
		float y=0.f, i=0.f, q=0.f, gamma=1.8f;
		for(int p=0; p<12; ++p) // 12 clock cycles per pixel.
		{
			// NES NTSC modulator (square wave between two voltage levels):
			float spot = lo_and_hi[bisqwit_wave(p,color)];

			// De-emphasis bits attenuate a part of the signal:
			if(pass==1)
			{
				if(((entry & 0x40) && bisqwit_wave(p,12))
					|| ((entry & 0x80) && bisqwit_wave(p, 4))
					|| ((entry &0x100) && bisqwit_wave(p, 8))) spot *= attenuation;
			}

			// Normalize:
			float v = (spot - black) / (white-black) / 12.f;

			// Ideal TV NTSC demodulator:
			y += v;
			i += v * std::cos(3.141592653 * p / 6);
			q += v * std::sin(3.141592653 * p / 6); // Or cos(... p-3 ... )
			// Note: Integrating cos() and sin() for p-0.5 .. p+0.5 range gives
			//       the exactly same result, scaled by a factor of 2*cos(pi/12).
		}

		// Convert YIQ into RGB according to FCC-sanctioned conversion matrix.

		int rt = bisqwit_clamp(255 * bisqwit_gammafix(y +  0.946882f*i +  0.623557f*q,gamma));
		int gt = bisqwit_clamp(255 * bisqwit_gammafix(y + -0.274788f*i + -0.635691f*q,gamma));
		int bt = bisqwit_clamp(255 * bisqwit_gammafix(y + -1.108545f*i +  1.709007f*q,gamma));

		if(pass==0) myr = rt, myg = gt, myb = bt;
		else
		{
			float rscale = (float)rt / myr;
			float gscale = (float)gt / myg;
			float bscale = (float)bt / myb;
			#define BCLAMP(x) ((x)<0?0:((x)>255?255:(x)))
			if(myr!=0) r = (u8)(BCLAMP(r*rscale));
			if(myg!=0) g = (u8)(BCLAMP(g*gscale));
			if(myb!=0) b = (u8)(BCLAMP(b*bscale));
		}
	}



}

int main() {
    for (int i = 0; i < 8; ++i)
        for (int p = 0; p < 64; ++p) {
            u8 r = palette[p].r, g = palette[p].g, b = palette[p].b;
            ApplyDeemphasisBisqwit(i * 64 + p, r, g, b);
            std::printf("%d %d %d\n", r, g, b);
        }
}
