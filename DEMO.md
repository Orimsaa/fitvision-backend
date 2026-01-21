# FitVision - Quick Demo Guide

## 🚀 วิธีใช้งาน

### **Option 1: Webcam (Real-time)**

```powershell
cd C:\fit\FitVision
python demo.py
# เลือก 1 → กล้อง webcam
```

**การใช้งาน:**
- ยืนหน้ากล้อง
- ทำท่า deadlift
- ระบบจะแสดง:
  - ✅ CORRECT FORM (สีเขียว)
  - ❌ INCORRECT FORM (สีแดง)
  - Confidence score
- กด `q` เพื่อออก

---

### **Option 2: วิดีโอไฟล์**

```powershell
python demo.py
# เลือก 2 → ใส่ path วิดีโอ
```

**ตัวอย่าง:**
```
Enter video path: C:\fit\Raw MP4 Videos\R_corr_1.mp4
```

**ผลลัพธ์:**
- แสดงวิดีโอพร้อม analysis
- สรุปท้ายวิดีโอ:
  - จำนวน frames ที่วิเคราะห์
  - % ท่าถูก/ผิด

---

## 📊 ตัวอย่างผลลัพธ์

```
ANALYSIS SUMMARY
============================================================
Total frames analyzed: 220
Correct form: 218 (99.1%)
Incorrect form: 2 (0.9%)
============================================================
```

---

## 💡 Tips

1. **แสงสว่าง** - ใช้ในที่แสงสว่างดี
2. **มุมกล้อง** - ถ่ายจากด้านข้าง (side view) ดีที่สุด
3. **ระยะห่าง** - ยืนห่างกล้องพอเห็นทั้งตัว
4. **เสื้อผ้า** - สวมเสื้อผ้าพอดีตัวเพื่อ pose detection ที่ดี

---

## 🎯 ทดสอบกับวิดีโอตัวอย่าง

```powershell
# Correct form
python demo.py
# เลือก 2
# ใส่: C:\fit\Raw MP4 Videos\R_corr_1.mp4

# Incorrect form (rounded back)
python demo.py  
# เลือก 2
# ใส่: C:\fit\Raw MP4 Videos\R_rb_1.mp4
```

---

**พร้อมใช้งานแล้วครับ!** 🎉
