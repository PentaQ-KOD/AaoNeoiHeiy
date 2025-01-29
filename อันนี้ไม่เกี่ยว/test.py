import google.generativeai as genai

# 🔹 ตั้งค่า API Key
genai.configure(api_key="AIzaSyAJXeEGOjpxlisJhC5YkDhZ5eHysgYtEfw")

# 🔹 เลือกโมเดล Gemini Pro
model = genai.GenerativeModel("gemini-pro")

# 🔹 ฟังก์ชันดึงข้อมูลสภาพอากาศ
def get_weather(city):
    return f"The weather in {city} is {result}"

# 🔹 สร้างข้อความของผู้ใช้
messages = [{"role": "user", "content": "What's the weather in Bangkok?"}]

# 🔹 เรียกใช้โมเดล Gemini
response = model.generate_content(messages[0]["content"])

# 🔹 ตรวจสอบว่าโมเดลต้องการเรียกฟังก์ชันหรือไม่
if "weather" in response.text.lower():
    city = "Bangkok"  # ดึงชื่อเมืองจากข้อความจริง
    result = get_weather(city)
else:
    result = response.text  # ถ้าไม่ใช่คำสั่งฟังก์ชัน ให้ใช้ข้อความโมเดล

print(result)
