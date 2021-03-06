/*
 * Copyright (C) 2015, Simon Fuhrmann, Ronny Klowsky
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 *
 * Notes:
 * - The implementation allows a minmum octave of -1 only.
 * - The descriptor extration supports 128 dimensions only.
 * - Coordinates in the keypoint are relative to the octave.
 *   Absolute coordinates are obtained by (TODO why? explain):
 *   (x + 0.5, y + 0.5) * 2^octave - (0.5, 0.5).
 * - Memory consumption is quite high, especially with large images.
 *   TODO: Find a more efficent code path to create octaves.
 */
#ifndef SFM_SIFT_HEADER
#define SFM_SIFT_HEADER

#include <string>
#include <vector>

#include "math/vector.h"
#include "core/image.h"
#include "features/defines.h"

namespace features {

    /**
     * Implementation of the SIFT feature detector and descriptor.
     * The implementation follows the description of the journal article:
     *
     *   Distinctive Image Features from Scale-Invariant Keypoints,
     *   David G. Lowe, International Journal of Computer Vision, 2004.
     *
     * The implementation used the siftpp implementation as reference for some
     * parts of the algorithm. This implementation is available here:
     *
     *   http://www.vlfeat.org/~vedaldi/code/siftpp.html
     */
    class Sift
    {
    public:
        /**
         * SIFT options.
         */
        struct Options
        {
            Options(void);

            int num_samples_per_octave;

            int min_octave;

            int max_octave;

            float contrast_threshold;

            float edge_ratio_threshold;
            float base_blur_sigma;

            /**
             * Sets the inherent blur sigma in the input image. Default is 0.5.
             * This is a technical detail and can usually be left alone.
             */
            float inherent_blur_sigma;

            /**
             * Produce status messages on the console.
             */
            bool verbose_output;

            /**
             * Produce even more messages on the console.
             */
            bool debug_output;
        };

        /**
         * Representation of a SIFT keypoint.
         * The keypoint locations are relative to the resampled size in
         * the image pyramid. To get the size relative to the input image,
         * each of (ix,iy,x,y) need to be multiplied with 2^o, where o
         * is the octave index of the keypoint. The octave index is -1 for the
         * upsampled image, 0 for the input image and >0 for subsampled images.
         * Note that the scale of the KP is already relative to the input image.
         */
        struct Keypoint
        {
            /** Octave index of the keypoint. Can be negative. */
            int octave;
            /** Sample index. Initally integer in {0 ... S-1}, later in [-1,S]. */
            float sample;
            /** Keypoint x-coordinate. Initially integer, later sub-pixel. */
            float x;
            /** Keypoint y-coordinate. Initially integer, later sub-pixel. */
            float y;
        };

        /**
         * Representation of the SIFT descriptor.
         * The descriptor is created in a rotation invariant way. The resulting
         * vector is unsigned and normalized, and has 128 dimensions.
         */
        struct Descriptor
        {
            /** The sub-pixel x-coordinate of the image keypoint. */
            float x;
            /** The sub-pixel y-coordinate of the image keypoint. */
            float y;
            /** The scale (or sigma value) of the keypoint. */
            float scale;
            /** The orientation of the image keypoint in [0, 2PI]. */
            float orientation;
            /** The descriptor data, elements are unsigned in [0.0, 1.0]. */
            math::Vector<float, 128> data;
        };

    public:
        typedef std::vector<Keypoint> Keypoints;
        typedef std::vector<Descriptor> Descriptors;

    public:
        explicit Sift(Options const& options);

        /** Sets the input image. */
        void set_image(core::ByteImage::ConstPtr img);
        /** Sets the input image. */
        void set_float_image(core::FloatImage::ConstPtr img);

        /** Starts the SIFT keypoint detection and descriptor extraction. */
        void process(void);

        /** Returns the list of keypoints. */
        Keypoints const& get_keypoints(void) const;

        /** Returns the list of descriptors. */
        Descriptors const& get_descriptors(void) const;

        /**
         * Helper function that creates SIFT descriptors from David Lowe's
         * SIFT descriptor files.
         */
        static void load_lowe_descriptors(std::string const& filename,
            Descriptors* result);

    protected:
        /**
         * Representation of a SIFT octave.
         */
        struct Octave
        {
            typedef std::vector<core::FloatImage::Ptr> ImageVector;
            ImageVector img; ///< S+3 images per octave
            ImageVector dog; ///< S+2 difference of gaussian images
            ImageVector grad; ///< S+3 gradient images
            ImageVector ori; ///< S+3 orientation images
        };

    protected:
        typedef std::vector<Octave> Octaves;

    protected:
        void create_octaves(void);
        void add_octave(core::FloatImage::ConstPtr image,
            float has_sigma, float target_sigma);
        void extrema_detection(void);
        std::size_t extrema_detection(core::FloatImage::ConstPtr s[3],
            int oi, int si);
        void keypoint_localization(void);

        void descriptor_generation(void);
        void generate_grad_ori_images(Octave* octave);
        void orientation_assignment(Keypoint const& kp,
            Octave const* octave, std::vector<float>& orientations);
        bool descriptor_assignment(Keypoint const& kp, Descriptor& desc,
            Octave const* octave);

        float keypoint_relative_scale(Keypoint const& kp);
        float keypoint_absolute_scale(Keypoint const& kp);

    private:
        Options options;
        core::FloatImage::ConstPtr orig; // Original input image
        Octaves octaves; // The image pyramid (the octaves)
        Keypoints keypoints; // Detected keypoints
        Descriptors descriptors; // Final SIFT descriptors
    };

    /* ---------------------------------------------------------------- */

    inline
        Sift::Options::Options(void)
        : num_samples_per_octave(3)
        , min_octave(0)
        , max_octave(4)
        , contrast_threshold(-1.0f)
        , edge_ratio_threshold(10.0f)
        , base_blur_sigma(1.6f)
        , inherent_blur_sigma(0.5f)
        , verbose_output(false)
        , debug_output(false)
    {
    }

    inline Sift::Keypoints const&
        Sift::get_keypoints(void) const
    {
        return this->keypoints;
    }

    inline Sift::Descriptors const&
        Sift::get_descriptors(void) const
    {
        return this->descriptors;
    }

}

#endif /* SFM_SIFT_HEADER */
