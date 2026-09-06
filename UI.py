from flask import Flask, request, render_template_string
import joblib
import numpy as np
import pandas as pd
UI = Flask(__name__)
model = joblib.load("model1.pkl")
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Academic Risk Predictor</title>
</head>
<body style="margin:0; padding:0; background:linear-gradient(135deg, #667EEA 0%, #764BA2 45%, #F093FB 100%); font-family: Georgia, 'Times New Roman', serif; color:#1C2541; min-height:100vh;">

  <div style="max-width:640px; margin:0 auto; padding:48px 24px 80px 24px;">

    <div style="background-color:#FFFFFF; border-radius:16px; box-shadow:0 12px 30px rgba(0,0,0,0.25); padding:36px 32px;">

      <div style="border-bottom:3px solid #764BA2; padding-bottom:18px; margin-bottom:32px;">
        <h1 style="font-size:28px; font-weight:normal; letter-spacing:0.3px; margin:0 0 6px 0; color:#4A2E83;">🎓 Academic Risk Predictor</h1>
        <p style="font-family: Verdana, Geneva, sans-serif; font-size:13px; color:#6B5B95; margin:0; line-height:1.5;">Enter a student's current record to estimate their risk of academic difficulty this semester.</p>
      </div>

      {% if prediction is none %}
      <!-- FORM VIEW -->
      <form method="POST">

        <div style="margin-bottom:20px;">
          <label style="display:block; font-family: Verdana, Geneva, sans-serif; font-size:13px; color:#5B3E96; font-weight:bold; margin-bottom:6px;">Previous GPA (out of 10)</label>
          <input type="number" step="any" name="x1" min="0" max="10" required
            style="width:100%; box-sizing:border-box; padding:11px 12px; font-size:15px; font-family: Verdana, Geneva, sans-serif; border:2px solid #A78BDB; border-radius:8px; background-color:#F6F3FD; color:#1C2541; outline:none;">
        </div>

        <div style="margin-bottom:20px;">
          <label style="display:block; font-family: Verdana, Geneva, sans-serif; font-size:13px; color:#2E7DB8; font-weight:bold; margin-bottom:6px;">Attendance percentage (out of 100)</label>
          <input type="number" step="any" name="x2" min="0" max="100" required
            style="width:100%; box-sizing:border-box; padding:11px 12px; font-size:15px; font-family: Verdana, Geneva, sans-serif; border:2px solid #7FB8E0; border-radius:8px; background-color:#EFF7FC; color:#1C2541; outline:none;">
        </div>

        <div style="margin-bottom:20px;">
          <label style="display:block; font-family: Verdana, Geneva, sans-serif; font-size:13px; color:#1C8C6B; font-weight:bold; margin-bottom:6px;">Internals (out of 100)</label>
          <input type="number" step="any" name="x3" min="0" max="100" required
            style="width:100%; box-sizing:border-box; padding:11px 12px; font-size:15px; font-family: Verdana, Geneva, sans-serif; border:2px solid #7FD6B8; border-radius:8px; background-color:#EDFAF4; color:#1C2541; outline:none;">
        </div>

        <div style="margin-bottom:20px;">
          <label style="display:block; font-family: Verdana, Geneva, sans-serif; font-size:13px; color:#C2410C; font-weight:bold; margin-bottom:6px;">Backlogs</label>
          <input type="number" step="any" name="x4" min="0" max="50" required
            style="width:100%; box-sizing:border-box; padding:11px 12px; font-size:15px; font-family: Verdana, Geneva, sans-serif; border:2px solid #F0A87A; border-radius:8px; background-color:#FDF3EC; color:#1C2541; outline:none;">
        </div>

        <div style="margin-bottom:20px;">
          <label style="display:block; font-family: Verdana, Geneva, sans-serif; font-size:13px; color:#B8860B; font-weight:bold; margin-bottom:6px;">Study hours (per week)</label>
          <input type="number" step="any" name="x5" min="0" max="130" required
            style="width:100%; box-sizing:border-box; padding:11px 12px; font-size:15px; font-family: Verdana, Geneva, sans-serif; border:2px solid #EBCB7A; border-radius:8px; background-color:#FDF9EC; color:#1C2541; outline:none;">
        </div>

        <div style="margin-bottom:28px;">
          <label style="display:block; font-family: Verdana, Geneva, sans-serif; font-size:13px; color:#C2185B; font-weight:bold; margin-bottom:6px;">Courses (current semester)</label>
          <input type="number" step="any" name="x6" min="0" max="10" required
            style="width:100%; box-sizing:border-box; padding:11px 12px; font-size:15px; font-family: Verdana, Geneva, sans-serif; border:2px solid #F2A0C4; border-radius:8px; background-color:#FDEFF5; color:#1C2541; outline:none;">
        </div>

        <button type="submit"
          style="width:100%; padding:15px; font-size:15px; font-weight:bold; font-family: Verdana, Geneva, sans-serif; letter-spacing:0.5px; background:linear-gradient(135deg, #667EEA, #764BA2); color:#FFFFFF; border:none; border-radius:10px; cursor:pointer; box-shadow:0 6px 16px rgba(118,75,162,0.4);">
          🔮 Predict
        </button>
      </form>

      {% else %}
      <!-- RESULT VIEW -->
      <div>

        <div style="display:flex; justify-content:center; margin:8px 0 30px 0;">
          <svg width="220" height="220" viewBox="0 0 220 220">
            <circle cx="110" cy="110" r="95" fill="none" stroke="#EDE9F7" stroke-width="16"></circle>
            <circle cx="110" cy="110" r="95" fill="none" stroke="{{ color }}" stroke-width="16"
              stroke-linecap="round" stroke-dasharray="596.9" stroke-dashoffset="{{ offset }}"
              transform="rotate(-90 110 110)"></circle>
            <text x="110" y="103" text-anchor="middle"
              style="font-family: Georgia, serif; font-size:36px; font-weight:bold; fill:{{ color }};">{{ prediction }}%</text>
            <text x="110" y="130" text-anchor="middle"
              style="font-family: Verdana, sans-serif; font-size:13px; letter-spacing:0.5px; fill:#6B5B95;">{{ level }}</text>
          </svg>
        </div>

        <div style="text-align:center; margin-bottom:30px; padding:14px; font-family: Verdana, Geneva, sans-serif; font-size:14px; border-radius:10px; background-color:{{ banner_bg }}; color:{{ banner_text }};">
          {{ message }}
        </div>

        <div style="border-top:3px solid #E4DAFA; padding-top:20px;">
          <h2 style="font-size:16px; font-weight:normal; margin:0 0 16px 0; color:#4A2E83;">📋 Submitted details</h2>

          <table style="width:100%; border-collapse:collapse; font-family: Verdana, Geneva, sans-serif; font-size:13px;">
            <tr>
              <td style="padding:10px 8px; color:#6B5B95; background-color:#F9F7FD; border-radius:6px 0 0 6px;">Previous GPA</td>
              <td style="padding:10px 8px; text-align:right; color:#1C2541; font-weight:bold; background-color:#F9F7FD; border-radius:0 6px 6px 0;">{{ x1 }} / 10</td>
            </tr>
            <tr><td colspan="2" style="height:6px;"></td></tr>
            <tr>
              <td style="padding:10px 8px; color:#2E7DB8; background-color:#EFF7FC; border-radius:6px 0 0 6px;">Attendance percentage</td>
              <td style="padding:10px 8px; text-align:right; color:#1C2541; font-weight:bold; background-color:#EFF7FC; border-radius:0 6px 6px 0;">{{ x2 }} %</td>
            </tr>
            <tr><td colspan="2" style="height:6px;"></td></tr>
            <tr>
              <td style="padding:10px 8px; color:#1C8C6B; background-color:#EDFAF4; border-radius:6px 0 0 6px;">Internals</td>
              <td style="padding:10px 8px; text-align:right; color:#1C2541; font-weight:bold; background-color:#EDFAF4; border-radius:0 6px 6px 0;">{{ x3 }} / 100</td>
            </tr>
            <tr><td colspan="2" style="height:6px;"></td></tr>
            <tr>
              <td style="padding:10px 8px; color:#C2410C; background-color:#FDF3EC; border-radius:6px 0 0 6px;">Backlogs</td>
              <td style="padding:10px 8px; text-align:right; color:#1C2541; font-weight:bold; background-color:#FDF3EC; border-radius:0 6px 6px 0;">{{ x4 }}</td>
            </tr>
            <tr><td colspan="2" style="height:6px;"></td></tr>
            <tr>
              <td style="padding:10px 8px; color:#B8860B; background-color:#FDF9EC; border-radius:6px 0 0 6px;">Study hours per week</td>
              <td style="padding:10px 8px; text-align:right; color:#1C2541; font-weight:bold; background-color:#FDF9EC; border-radius:0 6px 6px 0;">{{ x5 }} hrs/week</td>
            </tr>
            <tr><td colspan="2" style="height:6px;"></td></tr>
            <tr>
              <td style="padding:10px 8px; color:#C2185B; background-color:#FDEFF5; border-radius:6px 0 0 6px;">Courses this semester</td>
              <td style="padding:10px 8px; text-align:right; color:#1C2541; font-weight:bold; background-color:#FDEFF5; border-radius:0 6px 6px 0;">{{ x6 }}</td>
            </tr>
          </table>
        </div>

        <form method="GET">
          <button type="submit"
            style="width:100%; margin-top:28px; padding:14px; font-size:15px; font-weight:bold; font-family: Verdana, Geneva, sans-serif; letter-spacing:0.5px; background:linear-gradient(135deg, #F093FB, #F5576C); color:#FFFFFF; border:none; border-radius:10px; cursor:pointer; box-shadow:0 6px 16px rgba(245,87,108,0.4);">
            🔁 Check another student
          </button>
        </form>
      </div>
      {% endif %}

    </div>
  </div>
</body>
</html>
"""

@UI.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    offset=None
    circumference=None
    color=None
    banner_bg=None
    banner_text=None
    message=None
    level=None
    x1=None
    x2=None
    x3=None
    x4=None
    x5=None
    x6=None
    if request.method == "POST":
        x1 = float(request.form["x1"])
        x2 = float(request.form["x2"])
        x3 = float(request.form["x3"])
        x4 = float(request.form["x4"])
        x5 = float(request.form["x5"])
        x6 = float(request.form["x6"])
        features = pd.DataFrame([[x1, x2, x3, x4, x5, x6]],columns=['PREV_GPA','ATTENDANCE','INTERNALS','BACKLOGS','STUDY_HRS_PER_WEEK','CURRENT_COURSE_LOAD'])
        prediction = model.predict(features)[0]
        prediction=round(prediction,2)
        if prediction<0:
            prediction=0
        elif prediction>100:
            prediction=100
        circumference = 596.9
        offset = circumference-(prediction/100)*circumference

        if prediction < 34:
            color, level = "#2E8B57", "LOW RISK"
            banner_bg, banner_text = "#E1F7EC", "#1C7A4E"
            message = "✅ This student shows strong indicators. Continue current habits."
        elif prediction < 67:
            color, level = "#E8A33D", "MODERATE RISK"
            banner_bg, banner_text = "#FDF0DA", "#9A5F0B"
            message = "⚠️ This student shows some warning signs. Recommend closer monitoring."
        else:
            color, level = "#F5576C", "HIGH RISK"
            banner_bg, banner_text = "#FDE3E7", "#B8203A"
            message = "🚨 This student shows significant warning signs. Recommend immediate academic support."

    return render_template_string(HTML_TEMPLATE, prediction=prediction, offset=offset,
                            color=color, level=level, message=message,
                            banner_bg=banner_bg, banner_text=banner_text,
                            x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6)

if __name__ == "__main__":
    UI.run(debug=True)