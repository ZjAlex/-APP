package org.tensorflow.demo;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import org.tensorflow.demo.MtcnnAlignActivity;


public class MainActivity extends Activity{

    private Button face_detection;
    private Button face_rank;
    private Button quit;
    private Button face_align;
    private Button face_align_detect;
    private Button face_ver;
    private Button face_cls;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Set up the UI.
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        quit = (Button)findViewById(R.id.m_quit);
        quit.setTextSize(20);
        quit.setTextColor(Color.RED);
        quit.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        moveTaskToBack(true);
                        android.os.Process.killProcess(android.os.Process.myPid());
                        System.exit(1);
                    }
                }
        );
        quit.getBackground().setAlpha(0);
        face_detection = (Button)findViewById(R.id.m_fd);
        face_detection.setTextColor(Color.parseColor("#FF8888"));
        face_detection.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        start_detect(v);
                    }
                }
        );
        face_detection.setTextSize(20);
        face_detection.getBackground().setAlpha(0);

        face_rank = (Button)findViewById(R.id.m_rank);
        face_rank.setTextColor(Color.parseColor("#FF8888"));
        face_rank.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        start_rank(v);
                    }
                }
        );
        face_rank.setTextSize(20);
        face_rank.getBackground().setAlpha(0);

        face_align = (Button)findViewById(R.id.m_align);
        face_align.setTextColor(Color.parseColor("#FF8888"));
        face_align.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        start_align(v);
                    }
                }
        );
        face_align.setTextSize(20);
        face_align.getBackground().setAlpha(0);

        face_align_detect = (Button)findViewById(R.id.m_align_detect);
        face_align_detect.setTextColor(Color.parseColor("#FF8888"));
        face_align_detect.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        start_align_detect(v);
                    }
                }
        );
        face_align_detect.setTextSize(20);
        face_align_detect.getBackground().setAlpha(0);

        face_ver = (Button)findViewById(R.id.m_face_ver);
        face_ver.setTextColor(Color.parseColor("#FF8888"));
        face_ver.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        start_face_ver(v);
                    }
                }
        );
        face_ver.setTextSize(20);
        face_ver.getBackground().setAlpha(0);

        face_ver = (Button)findViewById(R.id.m_face_cls);
        face_ver.setTextColor(Color.parseColor("#FF8888"));
        face_ver.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        start_face_cls(v);
                    }
                }
        );
        face_ver.setTextSize(20);
        face_ver.getBackground().setAlpha(0);

    }

    public void start_detect(View view)
    {
        Intent intent = new Intent(this, DetectorActivity.class);
        startActivity(intent);
    }
    public void start_rank(View view)
    {
        Intent intent = new Intent(this,RankActivity.class);
        startActivity(intent);
    }
    public void start_align(View view)
    {
        Intent intent = new Intent(this,MtcnnAlignActivity.class);
        startActivity(intent);
    }
    public void start_align_detect(View view)
    {
        Intent intent = new Intent(this,MtcnnDetectorActivity.class);
        startActivity(intent);
    }

    public void start_face_ver(View view)
    {
        Intent intent = new Intent(this,FaceActivity.class);
        startActivity(intent);
    }
    public void start_face_cls(View view)
    {
        Intent intent = new Intent(this,FaceClassifierActivity.class);
        startActivity(intent);
    }
}
