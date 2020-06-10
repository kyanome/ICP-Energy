/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#include <QApplication>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include "object3d.h"
#include "pose_estimator6d.h"

using namespace std;
using namespace cv;


pcl::PointCloud<pcl::PointXYZ>::Ptr vector_to_pcl_pointcloud(vector<cv::Vec3f> verts)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud (new pcl::PointCloud<pcl::PointXYZ>);
    
    for(int i=0; i<verts.size(); i++)
    {
        pcl::PointXYZ point;
        point.x = verts[i][0];
        point.y = verts[i][1];
        point.z = verts[i][2];
        ptCloud->points.push_back(point);
    }
    return ptCloud;
}





void parseCSV(std::vector<std::vector<std::string>> &parsedCsv, string file_name)
{
    std::ifstream  data(file_name);
    std::string line;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        while(std::getline(lineStream,cell, ','))
        {
            parsedRow.push_back(cell);
        }

        parsedCsv.push_back(parsedRow);
    }
};


cv::Vec3f depth2point_cloud(const Matx33f K,  const Mat depth, cv::Vec3f verts) {
    float focal_length_x = K(0, 0);
    float focal_length_y = K(1, 1);
    float principal_point_x = K(0, 2);
    float principal_point_y = K(1, 2);

    float x_c = verts[0];
    float y_c = verts[1];
    float z_c = verts[2];

    float pixel_x = (x_c/z_c)*focal_length_x + principal_point_x;
    float pixel_y = (y_c/z_c)*focal_length_y + principal_point_y;

    float depth_value = depth.at<unsigned short>(pixel_y, pixel_x); 
    float x = (pixel_x - principal_point_x)*depth_value / focal_length_x;
    float y = (pixel_y - principal_point_y)*depth_value / focal_length_y;

    return cv::Vec3f(x, y, depth_value);

}


void get_destination_verts(const Mat depth, const vector<Object3D*>& objects, const Matx33f K, vector<cv::Vec3f> &source_points, vector<cv::Vec3f> &dest_points, vector<cv::Vec3f> &normal_points)
{
    vector<cv::Vec3f> model_verts = objects[0]->getVertices();
    vector<cv::Vec3f> normal_verts = objects[0]->getNormals();

    cv::Matx44f poseData = objects[0]->getPose() * objects[0]->getNormalization();
    cv::Matx33f R = cv::Matx33f(poseData(0,0), poseData(0,1), poseData(0,2), 
                        poseData(1,0), poseData(1,1), poseData(1,2),
                        poseData(2,0), poseData(2,1), poseData(2,2));
    cv::Vec3f  trans = cv::Vec3f(poseData(0,3), poseData(1,3), poseData(2,3));


    for(int i=0; i< model_verts.size()-1; i++){
        float sx =  cv::Vec3f(R(0,0), R(0,1), R(0,2)).dot(model_verts[i]) + trans[0]; 
        float sy =  cv::Vec3f(R(1,0), R(1,1), R(1,2)).dot(model_verts[i]) + trans[1];
        float sz =  cv::Vec3f(R(2,0), R(2,1), R(2,2)).dot(model_verts[i]) + trans[2];
        cv::Vec3f s_bar = cv::Vec3f(sx, sy, sz);

        cv::Vec3f _dest_points = depth2point_cloud(K, depth, s_bar);
        float dx = _dest_points[0];
        float dy = _dest_points[1];
        float dz = _dest_points[2];

        float nx =  cv::Vec3f(R(0,0), R(0,1), R(0,2)).dot(normal_verts[i]); 
        float ny =  cv::Vec3f(R(1,0), R(1,1), R(1,2)).dot(normal_verts[i]);
        float nz =  cv::Vec3f(R(2,0), R(2,1), R(2,2)).dot(normal_verts[i]);
        cv::Vec3f n_bar = cv::Vec3f(nx, ny, nz);

        //remove occulision points
        if ( (dz+10.0) < sz || 610<dz )
        {
            continue;
        }

        else
        {
            source_points.push_back(model_verts[i]);
            dest_points.push_back(_dest_points);
            normal_points.push_back(normal_verts[i]);
        }
    }

}

cv::Mat drawResultOverlay(const vector<Object3D*>& objects, const cv::Mat& frame)
{
    // render the models with phong shading
    RenderingEngine::Instance()->setLevel(0);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(1.0, 0.5, 0.0));
    //colors.push_back(Point3f(0.2, 0.3, 1.0));
    RenderingEngine::Instance()->renderShaded(vector<Model*>(objects.begin(), objects.end()), GL_FILL, colors, true);
    
    // download the rendering to the CPU
    Mat rendering = RenderingEngine::Instance()->downloadFrame(RenderingEngine::RGB);
    
    // download the depth buffer to the CPU
    Mat depth = RenderingEngine::Instance()->downloadFrame(RenderingEngine::DEPTH);
    
    // compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
    Mat result = frame.clone();
    for(int y = 0; y < frame.rows; y++)
    {
        for(int x = 0; x < frame.cols; x++)
        {
            Vec3b color = rendering.at<Vec3b>(y,x);
            if(depth.at<float>(y,x) != 0.0f)
            {
                result.at<Vec3b>(y,x)[0] = color[2];
                result.at<Vec3b>(y,x)[1] = color[1];
                result.at<Vec3b>(y,x)[2] = color[0];
            }
        }
    }
    return result;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // camera image size
    int width = 640;
    int height = 480;
    
    // near and far plane of the OpenGL view frustum
    float zNear = 10.0;
    float zFar = 10000.0;
    
    // camera instrinsics
    Matx33f K = Matx33f(612, 0, 310, 0, 612, 243, 0, 0, 1);
    Matx14f distCoeffs =  Matx14f(0.0, 0.0, 0.0, 0.0);
    
    // distances for the pose detection template generation
    vector<float> distances = {200.0f, 400.0f, 600.0f};
    
    // load 3D objects
    vector<Object3D*> objects;
    objects.push_back(new Object3D("data/bunny2.obj", 0, 0, 550, -160, 0, 0, 1.0, 0.55f, distances));
    //objects.push_back(new Object3D("data/a_second_model.obj", -50, 0, 600, 30, 0, 180, 1.0, 0.55f, distances2));
    

    // create the pose estimator
    PoseEstimator6D* poseEstimator = new PoseEstimator6D(width, height, zNear, zFar, K, distCoeffs, objects);
    
    // move the OpenGL context for offscreen rendering to the current thread, if run in a seperate QT worker thread (unnessary in this example)
    //RenderingEngine::Instance()->getContext()->moveToThread(this);
    
    // active the OpenGL context for the offscreen rendering engine during pose estimation
    RenderingEngine::Instance()->makeCurrent();
    
    int timeout = 0;
    
    bool showHelp = true;

    Mat frame;
    int MAX_ITER = 30;


    // get destination points
    std::ostringstream oss;
    std::vector<std::vector<std::string>> parsedCsv;
    Mat depth_frame(480, 640, CV_16UC1);
    oss << "./data/" << "depth_frame_eval.csv" << std::flush;
    parseCSV(parsedCsv, oss.str()); 
    oss.str("");
    for(int i=0; i<parsedCsv.size(); i++){
        for(int j=0; j<640; j++){
            float depth = std::stof(parsedCsv[i][j].c_str());
            depth_frame.at<unsigned short>(i, j) = depth;
        }
    }

    vector<cv::Vec3f> source_points;
    vector<cv::Vec3f> dest_points;
    vector<cv::Vec3f> normal_points;
    get_destination_verts(depth_frame, objects, K, source_points, dest_points, normal_points);

    // check vert size.
    vector<vector<cv::Vec3f>> icp_verts = {source_points, dest_points, normal_points};
    cout << "icp_verts[0].size()" << icp_verts[0].size() << endl;
    cout << "icp_verts[1].size()" << icp_verts[1].size() << endl;
    cout << "icp_verts[2].size()" << icp_verts[2].size() << endl;

    // convert vector to point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud = vector_to_pcl_pointcloud(icp_verts[0]);
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(point_cloud);
    while (!viewer.wasStopped())
    {
    }
    
    // estimate pose.
    poseEstimator->estimatePoses_icp(icp_verts);

  
    // deactivate the offscreen rendering OpenGL context
    RenderingEngine::Instance()->doneCurrent();
    
    // clean up
    RenderingEngine::Instance()->destroy();
    
    for(int i = 0; i < objects.size(); i++)
    {
        delete objects[i];
    }
    objects.clear();
    
    delete poseEstimator;
}
