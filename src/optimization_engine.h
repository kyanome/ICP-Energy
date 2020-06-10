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

#ifndef OPTIMIZATION_ENGINE
#define OPTIMIZATION_ENGINE

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "rendering_engine.h"
#include "signed_distance_transform2d.h"
#include "tclc_histograms.h"
#include "object3d.h"

/**
 *  This class implements an iterative Gauss-Newton optimization strategy for
 *  minimizing the region-based cost function with respect to the 6DOF
 *  pose of multiple rigid 3D objects on the basis of tclc-histograms
 *  for pixel-wise posterior segmentation of a camera frame.
 */
class OptimizationEngine
{
public:
    /**
     *  Constructor of the optimization engine, that create a signed
     *  distance transform object for internal use.
     *
     *  @param width  The width in pixels of the camera frame at full resolution.
     *  @param height  The height in pixels of the camera frame at full resolution.
     */
    OptimizationEngine(int width, int height);
    
    ~OptimizationEngine();
    
    /**
     *  Performs an hierachical iterative Gauss-Newton pose optimization
     *  for multiple 3D objects based on a region-based cost fuction. The
     *  implementation is parallelized on the CPU and uses the GPU only
     *  for rendering the models with OpenGL. Given a coarse to fine image
     *  pyramid (with at least 3 levels, created with a scaling factor of 2)
     *  of the current camera frame, the poses of all provided 3D objects
     *  that have been initialized beforehand will be refined.
     *
     *  @param  imagePyramid A coarse to fine image pyramid of the camera frame showing the objects in question (at least 3 levels, RGB, uchar).
     *  @param  objects A collection 3d objects of which the poses are supposed to be optimized.
     *  @param  runs A factor specifiyng how many times the default number of iterations per level are supposed to be performed (default = 1).
     */
    void minimize_icp(std::vector<Object3D*> &objects, std::vector<std::vector<cv::Vec3f>> &icp_verts);
    void minimize(std::vector<cv::Mat> &imagePyramid, std::vector<Object3D*> &objects, int runs = 1);
    
private:
    static OptimizationEngine *instance;
    
    RenderingEngine *renderingEngine;
    
    SignedDistanceTransform2D *SDT2D;
    
    int width;
    int height;

    void runIteration_icp(std::vector<Object3D*>& objects, std::vector<std::vector<cv::Vec3f>> &icp_verts);
    void runIteration(std::vector<Object3D*> &objects, const std::vector<cv::Mat> &imagePyramid, int level);
    
    void computeJacobians_icp(Object3D *object,  std::vector<std::vector<cv::Vec3f>> &icp_verts, cv::Matx66f &JTJ, cv::Matx61f &JT);
    void parallel_computeJacobians(Object3D *object, const cv::Mat &frame, const cv::Mat &depth, const cv::Mat &depthInv, const cv::Mat &sdt, const cv::Mat &xyPos, const cv::Rect &roi, const cv::Mat &mask, int m_id, int level, cv::Matx66f &wJTJ, cv::Matx61f &JT, int threads);
    
    cv::Rect compute2DROI(Object3D *object, const cv::Size &maxSize, int offset);

    void applyStepGaussNewton_icp(Object3D* object, const cv::Matx66f& JTJ, const cv::Matx61f& JT);
    void applyStepGaussNewton(Object3D *object, const cv::Matx66f &wJTJ, const cv::Matx61f &JT);
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, the Jacobian terms required for
 *  the Gauss-Newton pose update step are computed for a single object.
 */
class Parallel_For_computeJacobiansGN: public cv::ParallelLoopBody
{
private:
    uchar *frameData, *maskData, *initializedData;
    
    float *histogramsFGData, *histogramsBGData, *sdtData, *depthData, *depthInvData, *K_invData;
    
    int *xyPosData;
    
    cv::Mat localFG, localBG;
    
    std::vector<cv::Point3i> centersIDs;
    
    int numHistograms, radius2, upscale, numBins, binShift, fullWidth, fullHeight, _m_id;
    
    float _fx, _fy, _zNear, _zFar;
    
    bool maskAvailable;
    
    cv::Rect _roi;
    
    cv::Matx33f K_inv;
    
    cv::Matx66f *_wJTJCollection;
    cv::Matx61f *_JTCollection;
    
    int _threads;
    
public:
    Parallel_For_computeJacobiansGN(TCLCHistograms *tclcHistograms, const cv::Mat &frame, const cv::Mat &sdt, const cv::Mat &xyPos, const cv::Mat &depth, const cv::Mat &depthInv, const cv::Matx33f &K, float zNear, float zFar, const cv::Rect &roi, const cv::Mat &mask, int m_id, int level, std::vector<cv::Matx66f> &wJTJCollection, std::vector<cv::Matx61f> &JTCollection, int threads)
    {
        frameData = frame.data;
        
        localFG = tclcHistograms->getLocalForegroundHistograms();
        localBG = tclcHistograms->getLocalBackgroundHistograms();
        
        histogramsFGData = (float*)localFG.ptr<float>();
        histogramsBGData = (float*)localBG.ptr<float>();
        
        centersIDs = tclcHistograms->getCentersAndIDs();
        
        initializedData = tclcHistograms->getInitialized().data;
        
        numHistograms = (int)centersIDs.size();
        
        int radius = tclcHistograms->getRadius();
        
        radius2 = radius*radius;
        
        upscale = pow(2, level);
        
        numBins = tclcHistograms->getNumBins();
        
        binShift = 8 - log(numBins)/log(2);
        
        fullWidth = frame.cols;
        fullHeight = frame.rows;
        
        sdtData = (float*)sdt.ptr<float>();
        xyPosData = (int*)xyPos.ptr<int>();
        
        depthData = (float*)depth.ptr<float>();
        depthInvData = (float*)depthInv.ptr<float>();
        
        maskAvailable = false;
        if(m_id > 0)
        {
            maskData = (uchar*)mask.ptr<uchar>();
            maskAvailable = true;
            _m_id = m_id;
        }
        
        K_inv = K.inv();
        K_invData = K_inv.val;
        
        _fx = K(0, 0);
        _fy = K(1, 1);
        
        _zNear = zNear;
        _zFar = zFar;
        
        _roi = roi;
        
        _wJTJCollection = wJTJCollection.data();
        _JTCollection = JTCollection.data();
        
        _threads = threads;
    }
    
    bool isOccluded (int idx, float dist, float d) const
    {
        if(dist > 0)
        {
            uchar mVal = maskData[idx];
            if(mVal != 0 && mVal != _m_id)
            {
                float d2 = 1.0f - depthData[idx];
                if(d2 < d)
                {
                    return true;
                }
            }
        }
        else
        {
            int xPos = xyPosData[2*idx];
            int yPos = xyPosData[2*idx+1];
            
            if(xPos < 0 || yPos < 0)
                return false;
            
            int idx2 = yPos*_roi.width + xPos;
            
            float DsdtDx2 = (sdtData[idx2 + 1] - sdtData[idx2 - 1])/2.0f;
            float DsdtDy2 = (sdtData[idx2 + _roi.width] - sdtData[idx2 - _roi.width])/2.0f;
            
            int xoffset = DsdtDx2 >= 0 ? 1 : -1;
            int yoffset = DsdtDy2 >= 0 ? 1*_roi.width : -1*_roi.width;
            
            if(xPos > 1 && xPos < _roi.width-2 && yPos > 1 && yPos < _roi.height-2)
            {
                // check pixel in x-offset direction
                uchar mVal = maskData[idx2 + xoffset];
                if(mVal != 0 && mVal != _m_id)
                {
                    float d2 = 1.0f - depthData[idx2 + xoffset];
                    if(d2 < d)
                    {
                        return true;
                    }
                }
                // check pixel in y-offset direction
                mVal = maskData[idx2 + yoffset];
                if(mVal != 0 && mVal != _m_id)
                {
                    float d2 = 1.0f - depthData[idx2 + yoffset];
                    if(d2 < d)
                    {
                        return true;
                    }
                }
                // check pixel in x- and y-offset direction
                mVal = maskData[idx2 + yoffset + xoffset];
                if(mVal != 0 && mVal != _m_id)
                {
                    float d2 = 1.0f - depthData[idx2 + yoffset + xoffset];
                    if(d2 < d)
                    {
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        int range = _roi.height/_threads;
        
        int jStart = r.start*range;
        if(r.start == 0)
            jStart = 1;
        
        int jEnd = r.end*range;
        if(r.end == _threads)
        {
            jEnd = _roi.height-1;
        }
        
        float* wJTJ = (float*)_wJTJCollection[r.start].val;
        float* JT = (float*)_JTCollection[r.start].val;
        
        float s = 1.2f;
        float s2 = s*s;
        
        for(int j = jStart; j < jEnd; j++)
        {
            float J[6];
            
            int idx = j*_roi.width + 1;
            
            for(int i = 1; i < _roi.width-1; i++, idx++)
            {
                float dist = sdtData[idx];
                
                if(fabs(dist) <= 8.0f)
                {
                    // the smoothed Heaviside value for this signed distance
                    float heaviside = 1.0f/float(CV_PI)*(-atan(dist*s)) + 0.5f;
                    
                    // the corresponding smoothed dirac delta value
                    float dirac = (1.0f / float(CV_PI)) * (s/(dist*s2*dist + 1.0f));
                    
                    // compute the average foreground and background posterior
                    // probablities from the given set of tclc-histograms
                    int pIdx = (j+_roi.y) * fullWidth + i+_roi.x;
                    
                    // compute the histogram bin index from the pixel's color
                    int ru = (frameData[3*pIdx] >> binShift);
                    int gu = (frameData[3*pIdx+1] >> binShift);
                    int bu = (frameData[3*pIdx+2] >> binShift);
                    
                    int binIdx = (ru * numBins + gu) * numBins + bu;
                    
                    float pYFVal = 0;
                    float pYBVal = 0;
                    
                    int cnt = 0;
                    
                    for(int h = 0; h < numHistograms; h++)
                    {
                        cv::Point3i centerID = centersIDs[h];
                        
                        if(initializedData[centerID.z])
                        {
                            // check whether the pixel is within the local histogram region
                            int dx = centerID.x - upscale*(i+_roi.x + 0.5f);
                            int dy = centerID.y - upscale*(j+_roi.y + 0.5f);
                            int distance = dx*dx + dy*dy;
                            
                            if(distance <= radius2)
                            {
                                float pyf = localFG.at<float>(centerID.z, binIdx);
                                float pyb = localBG.at<float>(centerID.z, binIdx);
                                
                                pyf += 0.0000001f;
                                pyb += 0.0000001f;
                                
                                // compute local pixel-wise posteriors
                                pYFVal += pyf / (pyf + pyb);
                                pYBVal += pyb / (pyf + pyb);
                                
                                cnt++;
                            }
                        }
                    }
                    
                    if(cnt)
                    {
                        pYFVal /= cnt;
                        pYBVal /= cnt;
                    }
                    
                    // the energy inside the log
                    float e = heaviside * (pYFVal - pYBVal) + pYBVal + 0.000001;
                    
                    // the outer derivation
                    float DlogeDe = -(pYFVal - pYBVal) / e;
                    // the constant part of the overall gradient for this image
                    float constant_deriv = DlogeDe*dirac;
                    
                    float x = _roi.x;
                    float y = _roi.y;
                    float D;
                    
                    int zIdx;
                    
                    // get the closest pixel on the contour for pixels in the background
                    if(dist > 0)
                    {
                        int xPos = xyPosData[2*idx];
                        int yPos = xyPosData[2*idx+1];
                        
                        // should not happen
                        if(xPos < 0 || yPos < 0)
                            continue;
                        
                        x += xPos;
                        y += yPos;
                        zIdx = yPos*_roi.width + xPos;
                    }
                    else
                    {
                        x += i;
                        y += j;
                        zIdx = idx;
                    }
                    
                    // get the depth buffer value for this pixel
                    float depth = 1.0f - depthData[zIdx];
                    
                    // check for occlusions in case of multiple objects
                    if(maskAvailable && isOccluded(idx, dist, depth))
                        continue;
                    
                    // compute the Z-distance to the camera from the depth buffer value
                    D = 2.0f * _zNear * _zFar / (_zFar + _zNear - (2.0f*depth - 1.0) * (_zFar - _zNear));
                    
                    // back-project to camera coordinates
                    float X_c = D*(K_invData[0]*x+K_invData[2]);
                    float Y_c = D*(K_invData[4]*y+K_invData[5]);
                    float Z_c = D;
                    
                    float Z_c2 = Z_c*Z_c;
                    
                    // the image gradient of the signed distance transform
                    float DsdtDx = (sdtData[idx + 1] - sdtData[idx - 1])/2.0f;
                    float DsdtDy = (sdtData[idx + _roi.width] - sdtData[idx - _roi.width])/2.0f;

                    // compute the Jacobian of the signed distance transform with respect to
                    // the twist coordinates for this pixel
                    J[0] = DsdtDy*(-(_fy*pow(Y_c, 2))/Z_c2-_fy)-(DsdtDx*_fx*X_c*Y_c)/Z_c2;
                    J[1] = DsdtDx*((_fx*pow(X_c, 2))/Z_c2+_fx)+(DsdtDy*_fy*X_c*Y_c)/Z_c2;
                    J[2] = (DsdtDy*_fy*X_c)/Z_c-(DsdtDx*_fx*Y_c)/Z_c;
                    J[3] = (DsdtDx*_fx)/Z_c;
                    J[4] = (DsdtDy*_fy)/Z_c;
                    J[5] = -(DsdtDy*_fy*Y_c)/Z_c2-(DsdtDx*_fx*X_c)/Z_c2;
                    
                    // compute and add the per pixel gradient
                    for (int n = 0; n < 6; n++)
                    {
                        JT[n] += constant_deriv*J[n];
                    }
                    
                    float c2 = constant_deriv*constant_deriv;
                    
                    // compute the weighting term for this pixel
                    float w = -1.0f/log(e);
                    
                    // compute and add the per pixel Hessian approximation
                    for (int n = 0; n < 6; n++)
                    {
                        for (int m = n; m < 6; m++)
                        {
                            wJTJ[n * 6 + m] += w*J[n]*c2*J[m];
                        }
                    }
                    
                    // do the same for the inverse depth buffer
                    depth = 1.0f - depthInvData[zIdx];
                    
                    D = 2.0f * _zNear * _zFar / (_zFar + _zNear - (2.0f*depth - 1.0) * (_zFar - _zNear));
                    
                    X_c = D*(K_invData[0]*x+K_invData[2]);
                    Y_c = D*(K_invData[4]*y+K_invData[5]);
                    Z_c = D;
                    
                    Z_c2 = Z_c*Z_c;
                    
                    J[0] = DsdtDy*(-(_fy*pow(Y_c, 2))/Z_c2-_fy)-(DsdtDx*_fx*X_c*Y_c)/Z_c2;
                    J[1] = DsdtDx*((_fx*pow(X_c, 2))/Z_c2+_fx)+(DsdtDy*_fy*X_c*Y_c)/Z_c2;
                    J[2] = (DsdtDy*_fy*X_c)/Z_c-(DsdtDx*_fx*Y_c)/Z_c;
                    J[3] = (DsdtDx*_fx)/Z_c;
                    J[4] = (DsdtDy*_fy)/Z_c;
                    J[5] = -(DsdtDy*_fy*Y_c)/Z_c2-(DsdtDx*_fx*X_c)/Z_c2;
                    
                    for (int n = 0; n < 6; n++)
                    {
                        JT[n] += constant_deriv*J[n];
                    }
                    
                    for (int n = 0; n < 6; n++)
                    {
                        for (int m = n; m < 6; m++)
                        {
                            wJTJ[n * 6 + m] += w*J[n]*c2*J[m];
                        }
                    }
                }
            }
        }
    }
};



/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, the Jacobian terms required for
 *  the Gauss-Newton pose update step are computed for a single object.
 */
class computeJacobiansGN_icp: public cv::ParallelLoopBody
{
private:
    
    float *depthData;
    
    float focal_length_x, focal_length_y, principal_point_x, principal_point_y, _zNear, _zFar ;
    
    cv::Matx33f K_inv;
    cv::Matx14f distCoeff;

    cv::Matx44f poseData;
    cv::Matx33f R;
    cv::Vec3f trans;    

    std::vector<cv::Vec3f> model_verts;
    std::vector<cv::Vec3f> dest_points;
    std::vector<cv::Vec3f> normals;

    cv::Matx66f *_JTJCollection;
    cv::Matx61f *_JTCollection;

    
public:
    computeJacobiansGN_icp(Object3D* object, const std::vector<std::vector<cv::Vec3f>> &icp_verts, const cv::Matx33f &K, float zNear, float zFar, std::vector<cv::Matx66f> &JTJCollection, std::vector<cv::Matx61f> &JTCollection)
    {

        _JTJCollection = JTJCollection.data();
        _JTCollection = JTCollection.data();

        // intrinsic parameters
        focal_length_x = K(0, 0);
        focal_length_y = K(1, 1);
        principal_point_x = K(0, 2);
        principal_point_y = K(1, 2);

        distCoeff = cv::Matx14f(0.0, 0.0, 0.0, 0.0);

        _zNear = zNear;
        _zFar = zFar;
 
        // extrinsic parameters
        poseData = object->getPose() * object->getNormalization();
        R = cv::Matx33f(poseData(0,0), poseData(0,1), poseData(0,2), 
                        poseData(1,0), poseData(1,1), poseData(1,2),
                        poseData(2,0), poseData(2,1), poseData(2,2));
        trans = cv::Vec3f(poseData(0,3), poseData(1,3), poseData(2,3));

        // template shape
        model_verts  = icp_verts[0];
        dest_points  = icp_verts[1];
        normals      = icp_verts[2];
    }


    virtual void operator()( const cv::Range &r ) const
    {
        float J[6];
        float* JTJ = (float*)_JTJCollection[r.start].val;
        float* JT = (float*)_JTCollection[r.start].val;
        float all_residual;


        for(int i=0; i<model_verts.size(); i++)
        {
            float sx =  cv::Vec3f(R(0,0), R(0,1), R(0,2)).dot(model_verts[i]) + trans[0]; 
            float sy =  cv::Vec3f(R(1,0), R(1,1), R(1,2)).dot(model_verts[i]) + trans[1];
            float sz =  cv::Vec3f(R(2,0), R(2,1), R(2,2)).dot(model_verts[i]) + trans[2];
            cv::Vec3f s_bar = cv::Vec3f(sx, sy, sz);

            float dx = dest_points[i][0];
            float dy = dest_points[i][1];
            float dz = dest_points[i][2];

            float nx =  cv::Vec3f(R(0,0), R(0,1), R(0,2)).dot(normals[i]); 
            float ny =  cv::Vec3f(R(1,0), R(1,1), R(1,2)).dot(normals[i]);
            float nz =  cv::Vec3f(R(2,0), R(2,1), R(2,2)).dot(normals[i]);
            cv::Vec3f n_bar = cv::Vec3f(nx, ny, nz);

            float a1 = (  (nz*(sy - dy) - ny*(sz - dz))  +   ((nz*sy) - (ny*sz))  );
            float a2 = (  (nx*(sz - dz) - nz*(sx - dx))  +   ((nx*sz) - (nz*sx))  );
            float a3 = (  (ny*(sx - dx) - nx*(sy - dy))  +    ((ny*sx) - (nx*sy)) );
            float a4 = nx;
            float a5 = ny;
            float a6 = nz;

            float residual = n_bar.dot((s_bar - dest_points[i]));
            all_residual += residual;

            J[0] = a1;
            J[1] = a2;
            J[2] = a3;
            J[3] = a4;
            J[4] = a5;
            J[5] = a6;

            // compute and add the per pixel gradient
            for (int n = 0; n < 6; n++)
            {
                JT[n] += J[n]*residual;
            }
            
            // compute and add the per pixel Hessian approximation
            for (int n = 0; n < 6; n++)
            {
                for (int m = n; m < 6; m++)
                {
                    JTJ[n * 6 + m] += J[n]*J[m];
                }
            }

        }
        std::cout << "all residual = " << all_residual << std::endl;

    }
};


#endif //OPTIMIZATION_ENGINE
