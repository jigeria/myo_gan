package com.magmatart.videorecorder;

import android.content.Context;
import android.media.MediaRecorder;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;

/**
 * Created by Soomin Lee on 6/6/2018.
 */

public class CamcorderView extends SurfaceView implements
        SurfaceHolder.Callback {

    MediaRecorder recorder;
    SurfaceHolder holder;
    String outputFile = "/sdcard/default.mp4";

    public CamcorderView(Context context, AttributeSet attrs) {
        super(context, attrs);

        holder = getHolder();
        holder.addCallback(this);
        holder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setVideoSource(MediaRecorder.VideoSource.DEFAULT);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
        recorder.setVideoEncoder(MediaRecorder.VideoEncoder.MPEG_4_SP);
        // recorder.setVideoSize(480, 320);
        // recorder.setVideoFrameRate(15);
        // recorder.setMaxDuration(10000);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        recorder.setOutputFile(outputFile);
        recorder.setPreviewDisplay(holder.getSurface());
        if (recorder != null) {
            try {
                recorder.prepare();
            } catch (IllegalStateException e) {
                Log.e("IllegalStateException", e.toString());
            } catch (IOException e) {
                Log.e("IOException", e.toString());
            }
        }
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int width,
                               int height) {
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    public void setOutputFile(String filename)
    {
        outputFile = filename;
        recorder.setOutputFile(filename);
    }

    public void startRecording()
    {
        recorder.start();
    }

    public void stopRecording()
    {
        recorder.stop();
        recorder.release();
    }
}