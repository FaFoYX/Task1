#include <iostream>

#include "util/file_system.h"
#include "util/timer.h"
#include "core/image.h"
#include "core/image_tools.h"
#include "core/image_io.h"

#include "features/surf.h"
#include "features/sift.h"
#include "examples/task1/visualizer.h"

bool
sift_compare(features::Sift::Descriptor const& d1, features::Sift::Descriptor const& d2)
{
    return d1.scale > d2.scale;
}

int main7(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Syntax: " << argv[0] << " <image>" << std::endl;
        return 1;
    }

    /* ����ͼ��*/
    core::ByteImage::Ptr image;
    std::string image_filename = argv[1];
    try
    {
        std::cout << "Loading " << image_filename << "..." << std::endl;
        image = core::image::load_file(image_filename);
        //image = core::image::rescale_half_size<uint8_t>(image);
        //image = core::image::rescale_half_size<uint8_t>(image);
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    /* SIFT �������. */
    features::Sift::Descriptors sift_descr;
    features::Sift::Keypoints sift_keypoints;
    {
        features::Sift::Options sift_options;
        sift_options.verbose_output = true;
        sift_options.debug_output = true;
        features::Sift sift(sift_options);
        sift.set_image(image);

        util::WallTimer timer;
        sift.process(); // ������
        std::cout << "Computed SIFT features in "
            << timer.get_elapsed() << "ms." << std::endl;

        sift_descr = sift.get_descriptors();
        sift_keypoints = sift.get_keypoints();
    }

    // �������㰴�ճ߶Ƚ�������
    std::sort(sift_descr.begin(), sift_descr.end(), sift_compare);

    std::vector<features::Visualizer::Keypoint> sift_drawing;
    for (std::size_t i = 0; i < sift_descr.size(); ++i)
    {
        features::Visualizer::Keypoint kp;
        kp.orientation = sift_descr[i].orientation;
        kp.radius = sift_descr[i].scale;
        kp.x = sift_descr[i].x;
        kp.y = sift_descr[i].y;
        sift_drawing.push_back(kp);
    }

    core::ByteImage::Ptr sift_image = features::Visualizer::draw_keypoints(image,
        sift_drawing, features::Visualizer::RADIUS_BOX_ORIENTATION);

    /* ����ͼ���ļ��� */
    std::string sift_out_fname = "./tmp/" + util::fs::replace_extension
    (util::fs::basename(image_filename), "sift.png");
    core::image::save_file(sift_image, sift_out_fname);


    return 0;
}