/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import androidx.recyclerview.widget.RecyclerView
import com.google.firebase.FirebaseApp
import com.google.firebase.database.*
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfRect
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.objdetect.Objdetect
import org.tensorflow.lite.DataType
import org.tensorflow.lite.examples.classification.ui.RecognitionAdapter
import org.tensorflow.lite.examples.classification.util.YuvToRgbConverter
import org.tensorflow.lite.examples.classification.viewmodel.Recognition
import org.tensorflow.lite.examples.classification.viewmodel.RecognitionListViewModel
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

// Constants
private const val MAX_RESULT_DISPLAY = 3 // Maximum number of results displayed
private const val TAG = "TFL Classify" // Name for logging
private const val REQUEST_CODE_PERMISSIONS = 999 // Return code after asking for permission
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) // permission needed

// Listener for the result of the ImageAnalyzer
typealias RecognitionListener = (recognition: String) -> Unit

/**
 * Main entry point into TensorFlow Lite Classifier
 */
class MainActivity : AppCompatActivity() {


    private lateinit var mlistener: ValueEventListener
    private lateinit var mCascadeFile: File

    // CameraX variables
    private lateinit var preview: Preview // Preview use case, fast, responsive view of the camera
    private lateinit var imageAnalyzer: ImageAnalysis // Analysis use case, for running ML code
    private lateinit var camera: Camera
    private lateinit var mDatabase: DatabaseReference
    private  var u1:String = "1"
    private  var u2:String = "2"
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    // Views attachment
    private val emotionRes by lazy {
        findViewById<TextView>(R.id.recognitionResults) // Display the result of analysis
    }
    private val viewFinder by lazy {
        findViewById<PreviewView>(R.id.viewFinder) // Display the preview image from Camera
    }
    private val switch1 by lazy{
        findViewById<Switch>(R.id.switch1)
    }

    // Contains the recognition result. Since  it is a viewModel, it will survive screen rotations
    //private val recogViewModel: RecognitionListViewModel by viewModels()
    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }

        mlistener = object : ValueEventListener {
            override fun onDataChange(dataSnapshot: DataSnapshot) {
                val emotion = dataSnapshot.getValue(String::class.java)
                emotionRes.text = emotion;
            }

            override fun onCancelled(databaseError: DatabaseError) {
                // Getting Post failed, log a message
                Log.w(TAG, "loadPost:onCancelled", databaseError.toException())
                // ...
            }}
        mDatabase.child("emotion/$u1").addValueEventListener(mlistener)


    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        mDatabase = FirebaseDatabase.getInstance().reference

        switch1.setOnCheckedChangeListener { compoundButton, b ->

            mDatabase.child("emotion/$u1").removeEventListener(mlistener)

            if(switch1.isChecked){
                u1 = "1"
                u2 = "2"
            }else{
                u1 = "2"
                u2 = "1"
            }
            Log.d(TAG, "onCreate: dd" + "$u1    $u2")

            mlistener = object : ValueEventListener {
                override fun onDataChange(dataSnapshot: DataSnapshot) {
                    val emotion = dataSnapshot.getValue(String::class.java)
                    emotionRes.text = emotion;
                }

                override fun onCancelled(databaseError: DatabaseError) {
                    // Getting Post failed, log a message
                    Log.w(TAG, "loadPost:onCancelled", databaseError.toException())
                    // ...
                }}
            mDatabase.child("emotion/$u1").addValueEventListener(mlistener)

        }

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Initialising the resultRecyclerView and its linked viewAdaptor
        val viewAdapter = RecognitionAdapter(this)
//        resultRecyclerView.adapter = viewAdapter
//
//        // Disable recycler view animation to reduce flickering, otherwise items can move, fade in
//        // and out as the list change
//        resultRecyclerView.itemAnimator = null

        // Attach an observer on the LiveData field of recognitionList
        // This will notify the recycler view to update every time when a new list is set on the
        // LiveData field of recognitionList.
//        recogViewModel.recognitionList.observe(this,
//                Observer {
//                    val ar = ArrayList<Recognition>()
//                    ar.add(Recognition(it, 1f))
//
//                    viewAdapter.submitList(ar as List<Recognition>)
//                }
//        )

    }

    /**
     * Check all permissions are granted - use for Camera permission in this example.
     */
    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
                baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * This gets called after the Camera permission pop up is shown.
     */
    override fun onRequestPermissionsResult(
            requestCode: Int,
            permissions: Array<String>,
            grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                // Exit the app if permission is not granted
                // Best practice is to explain and offer a chance to re-request but this is out of
                // scope in this sample. More details:
                // https://developer.android.com/training/permissions/usage-notes
                Toast.makeText(
                        this,
                        getString(R.string.permission_deny_text),
                        Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    /**
     * Start the Camera which involves:
     *
     * 1. Initialising the preview use case
     * 2. Initialising the image analyser use case
     * 3. Attach both to the lifecycle of this activity
     * 4. Pipe the output of the preview object to the PreviewView on the screen
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            preview = Preview.Builder()
                    .build()

            imageAnalyzer = ImageAnalysis.Builder()
                    // This sets the ideal size for the image to be analyse, CameraX will choose the
                    // the most suitable resolution which may not be exactly the same or hold the same
                    // aspect ratio
                    .setTargetResolution(Size(224, 224))
                    // How the Image Analyser should pipe in input, 1. every frame but drop no frame, or
                    // 2. go to the latest frame and may drop some frame. The default is 2.
                    // STRATEGY_KEEP_ONLY_LATEST. The following line is optional, kept here for clarity
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analysisUseCase: ImageAnalysis ->
                        analysisUseCase.setAnalyzer(cameraExecutor, ImageAnalyzer(this) { items ->
                            // updating the list of recognised objects
                            //recogViewModel.updateData(items)
                        })
                    }

            // Select camera, back is the default. If it is not available, choose front camera
            val cameraSelector =
                    if (cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA))
                        CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera - try to bind everything at once and CameraX will find
                // the best combination.
                camera = cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalyzer
                )

                // Attach the preview to preview view, aka View Finder
                preview.setSurfaceProvider(viewFinder.surfaceProvider)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private inner class ImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :
            ImageAnalysis.Analyzer {
        private var faceCascade: CascadeClassifier = CascadeClassifier()

        init {

            try {
                // load cascade file from application resources
                val `is`: InputStream = resources.openRawResource(R.raw.haarcascade_frontalface_default)
                val cascadeDir: File = getDir("cascade", Context.MODE_PRIVATE)
                mCascadeFile = File(cascadeDir, "haarcascade_frontalface_default.xml")
                val os = FileOutputStream(mCascadeFile)
                val buffer = ByteArray(4096)
                var bytesRead: Int
                while (`is`.read(buffer).also { bytesRead = it } != -1) {
                    os.write(buffer, 0, bytesRead)
                }
                `is`.close()
                os.close()
                faceCascade = CascadeClassifier(mCascadeFile.absolutePath)
                if (faceCascade.empty()) {
                    Log.e(TAG, "Failed to load cascade classifier")
                } else Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath())

                cascadeDir.delete()
            } catch (e: IOException) {
                e.printStackTrace()
                Log.e(TAG, "Failed to load cascade. Exception thrown: $e")
            }
        }

        // TODO 1: Add class variable TensorFlow Lite Model
        // Initializing the flowerModel by lazy so that it runs in the same thread when the process
        // method is called.
        private val flowerModel: org.tensorflow.lite.examples.classification.ml.Model by lazy {

            // TODO 6. Optional GPU acceleration
            val compatList = CompatibilityList()

            val options = if (compatList.isDelegateSupportedOnThisDevice) {
                Log.d(TAG, "This device is GPU Compatible ")
                Model.Options.Builder().setDevice(Model.Device.GPU).build()
            } else {
                Log.d(TAG, "This device is GPU Incompatible ")
                Model.Options.Builder().setNumThreads(4).build()
            }

            // Initialize the Flower Model
            org.tensorflow.lite.examples.classification.ml.Model.newInstance(ctx, options)
        }

        override fun analyze(imageProxy: ImageProxy) {

            val labels = arrayOf("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
            val label_emoji = arrayOf("\uD83D\uDE21", "\uD83D\uDE12", "\uD83D\uDE31", "\uD83D\uDE01", "☹",
                    "\uD83D\uDE32", "\uD83D\uDE10")

            val label: String
            val emoji: String
            val i: Int


            // TODO 2: Convert Image to Bitmap then to TensorImage
            var bitmap = toBitmap(imageProxy)
            val source = Mat()
            Utils.bitmapToMat(bitmap, source)
            val faces = MatOfRect()
            val gray = Mat()

            Imgproc.cvtColor(source, gray, Imgproc.COLOR_RGB2GRAY)
            Log.d(TAG, "analyze: $gray   $source")
            faceCascade.detectMultiScale(gray, faces, 1.3, 5, Objdetect.CASCADE_FIND_BIGGEST_OBJECT)

            val faceArray = faces.toArray()

            if (faceArray.size != 0) {
                val cropped = Mat(source, faceArray[0])
                bitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped, bitmap)

                bitmap = bitmap?.let { Bitmap.createScaledBitmap(it, 48, 48, true) }
                // TODO 3: Process the image using the trained model, sort and pick out the top results
                val width = bitmap?.width
                val height = bitmap?.height
                val mImgData = ByteBuffer
                        .allocateDirect(4 * width!! * height!!)
                mImgData.order(ByteOrder.nativeOrder())
                val pixels = IntArray(width * height)
                bitmap?.getPixels(pixels, 0, width, 0, 0, width, height)
                for (pixel in pixels) {
                    mImgData.putFloat(Color.red(pixel).toFloat())
                }

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 48, 48, 1), DataType.FLOAT32)
                inputFeature0.loadBuffer(mImgData)

                val outputs = flowerModel.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer


                i = outputFeature0.floatArray.asList().indexOf(1f)
            } else {
                i = -1
            }

            if (i == -1) {
                label = "No Face"
                emoji = "None"
            } else {
                label = labels[i]
                emoji = label_emoji[i]
            }

            Log.d(TAG, "analyze: $label  $emoji")

            mDatabase.child("emotion/$u2").setValue("$label   $emoji")
            // Close the image,this tells CameraX to feed the next image to the analyzer
            imageProxy.close()
        }

        /**
         * Convert Image Proxy to Bitmap
         */
        private val yuvToRgbConverter = YuvToRgbConverter(ctx)
        private lateinit var bitmapBuffer: Bitmap
        private lateinit var rotationMatrix: Matrix

        @SuppressLint("UnsafeExperimentalUsageError")
        private fun toBitmap(imageProxy: ImageProxy): Bitmap? {

            val image = imageProxy.image ?: return null

            // Initialise Buffer
            if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                Log.d(TAG, "Initalise toBitmap()")
                rotationMatrix = Matrix()
                rotationMatrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                bitmapBuffer = Bitmap.createBitmap(
                        imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
                )
            }

            // Pass image to an image analyser
            yuvToRgbConverter.yuvToRgb(image, bitmapBuffer)

            // Create the Bitmap in the correct orientation
            return Bitmap.createBitmap(
                    bitmapBuffer,
                    0,
                    0,
                    bitmapBuffer.width,
                    bitmapBuffer.height,
                    rotationMatrix,
                    false
            )
        }

    }

}
