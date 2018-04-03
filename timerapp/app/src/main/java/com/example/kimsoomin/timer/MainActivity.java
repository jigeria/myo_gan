package com.example.kimsoomin.timer;

import android.content.Intent;
import android.media.AudioManager;
import android.media.SoundPool;
import android.os.SystemClock;
import android.os.Vibrator;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.FrameLayout;
import android.widget.TextView;

import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {

    public float time = 0;
    Button startbutton;
    Button stopbutton;
    TextView timer_text ;
    FrameLayout layout;
    String color = "FFFFFF";

    private Timer timer;
    private TimerTask timerTask;
    private  int stopflag = 0;
    private Chronometer cm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final long time = 2900;
        final long lastTime = System.currentTimeMillis();

        final SoundPool sp = new SoundPool(1, AudioManager.STREAM_MUSIC, 0);
        final int sound = sp.load(this, R.raw.beep, 1);

        final Vibrator vib = (Vibrator)getSystemService(VIBRATOR_SERVICE);

        timer_text = (TextView)findViewById(R.id.timer_text);
        cm = (Chronometer)findViewById(R.id.Time_show);

        startbutton = (Button)findViewById(R.id.startButton);
        startbutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cm.setBase(SystemClock.elapsedRealtime());
                cm.start();

                timer_text.setText("Timer is working");
                    timer = new Timer(true);
                    timerTask = new TimerTask() {
                        @Override
                        public void run() {
                            long currentTime = System.currentTimeMillis();
                            sp.play(sound, 1f, 1f, 0,0,1f);
                            vib.vibrate(200);

                            if(stopflag == 1){
                                timer.cancel();
                                cm.stop();
                            }
                        }
                    };
                    timer.schedule(timerTask,3000,3000);
            }
        });

        stopbutton = (Button)findViewById(R.id.stopButton);
        stopbutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopflag = 1;
                timer_text.setText("3 Second Timer");
            }
        });
    }
}

