import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
    
# 模型目录
model_dir = './models'

# 设置页面配置
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    nav = st.radio(
        "导航菜单",
        ["项目介绍", "专业数据分析", "测试成绩"]
    )

if nav == "项目介绍":
    # 标题
    st.title("📊 学生成绩分析与预测系统")
    st.divider()
    
    # 项目概述
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.header("📋 项目概述")
            st.write("学生成绩分析与预测系统是一个基于机器学习的数据分析平台，旨在帮助教育工作者从海量学生数据中提取有价值的信息，并通过预测模型为学生提供个性化的学习建议。")
            # 主要特点
            st.subheader("✨ 主要特点")
            st.write("- 📈 数据可视化：直观展示学生成绩分布和趋势")
            st.write("- 🤖 智能预测：基于历史数据预测学生未来成绩")
            st.write("- 🎯 个性化建议：针对每个学生提供定制化的学习建议")
            st.write("- 📊 多维分析：从多个维度分析学生学习情况")
        with col2:
            # 图片切换器
            st.subheader("项目展示")
            
            # 初始化会话状态，用于跟踪当前图片索引
            if 'current_image_index' not in st.session_state:
                st.session_state.current_image_index = 0
            
            # 图片列表和对应的图注
            image_data = [
                {"path": "index1.png", "caption": "项目展示图1：系统主界面"},
                {"path": "index2.png", "caption": "项目展示图2：数据分析可视化"},
                {"path": "index3.png", "caption": "项目展示图3：成绩预测功能"}
            ]
            
            # 显示当前图片
            current_image = image_data[st.session_state.current_image_index]
            st.image(current_image["path"], caption=current_image["caption"], width=500)
            
            # 切换按钮
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("上一张"):
                    st.session_state.current_image_index = (st.session_state.current_image_index - 1) % len(image_data)
            with col_next:
                if st.button("下一张"):
                    st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(image_data)

    st.divider()
    
    # 项目目标
    with st.container():
        st.header("🎯 项目目标")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 目标一")
            st.write("- 建立学生成绩数据库")
            st.write("- 实现数据可视化展示")
            st.write("- 提供多维度数据分析")
        
        with col2:
            st.subheader("🤖 目标二")
            st.write("- 构建成绩预测模型")
            st.write("- 实现智能推荐系统")
            st.write("- 提供个性化学习建议")
        
        with col3:
            st.subheader("🎯 目标三")
            st.write("- 优化教学管理流程")
            st.write("- 提升学生学习效率")
            st.write("- 辅助教育决策制定")
    
    st.divider()
    
    # 技术架构
    with st.container():
        st.header("🏗️ 技术架构")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("数据层")
            st.write("MySQL")
        
        with col2:
            st.subheader("处理层")
            st.write("Python")
            st.write("Pandas")
        
        with col3:
            st.subheader("模型层")
            st.write("Scikit-learn")
        
        with col4:
            st.subheader("展示层")
            st.write("Streamlit")

if nav == "专业数据分析":
    st.title("📊 专业数据分析")
    st.divider()
    
    # 读取真实数据
    data = pd.read_csv('./student_data_adjusted_rounded.csv')
    
    # 1. 各专业男女比例分析
    st.header("1. 各专业男女比例")
    
    # 计算各专业男女比例
    gender_major_data = data.groupby(['专业', '性别']).size().reset_index(name='人数')
    gender_major_pivot = gender_major_data.pivot(index='专业', columns='性别', values='人数').fillna(0)
    gender_major_pivot = gender_major_pivot.astype(int)
    
    # 创建柱形图
    fig1 = px.bar(gender_major_pivot, barmode="group", title="各专业男女比例")
    
    # 布局
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("性别比例数据")
        st.dataframe(gender_major_pivot, use_container_width=True)
    
    st.divider()
    
    # 2. 期中期末趋势指标对比
    st.header("2. 期中期末趋势指标对比")
    
    # 计算各专业期中期末成绩及每周学习时长平均值
    exam_trends = data.groupby('专业').agg({
        '期中考试分数': 'mean',
        '期末考试分数': 'mean',
        '每周学习时长（小时）': 'mean'
    }).reset_index()
    
    # 创建折线图
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=exam_trends['专业'], y=exam_trends['期中考试分数'], mode='lines+markers', name='期中考试分数'))
    fig2.add_trace(go.Scatter(x=exam_trends['专业'], y=exam_trends['期末考试分数'], mode='lines+markers', name='期末考试分数'))
    fig2.add_trace(go.Scatter(x=exam_trends['专业'], y=exam_trends['每周学习时长（小时）'], mode='lines+markers', name='每周学习时长（小时）'))
    fig2.update_layout(title="各专业期中期末成绩及学习时长趋势对比")
    
    # 布局
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("详细数据")
        st.dataframe(exam_trends, use_container_width=True)
    
    st.divider()
    
    # 3. 各专业出勤率分析
    st.header("3. 各专业出勤率分析")
    
    # 计算各专业平均出勤率
    attendance_data = data.groupby('专业').agg({
        '上课出勤率': 'mean'
    }).reset_index()
    attendance_data['上课出勤率'] = attendance_data['上课出勤率'] * 100
    attendance_data = attendance_data.sort_values(by='上课出勤率', ascending=False)
    attendance_data = attendance_data.rename(columns={'上课出勤率': '出勤率'})
    
    # 创建柱形图
    fig3 = px.bar(attendance_data, x="专业", y="出勤率", color="出勤率", color_continuous_scale="Viridis", title="各专业出勤率对比")
    
    # 布局
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.subheader("出勤率排名")
        st.dataframe(attendance_data, use_container_width=True)
    
    st.divider()
    
    # 4. 大数据管理专业专项分析
    st.header("4. 大数据管理专业专项分析")
    
    # 筛选大数据管理专业的数据
    bigdata_data = data[data['专业'] == '大数据管理']
    
    # 计算大数据管理专业各指标的分布
    bigdata_study_dist = bigdata_data[['每周学习时长（小时）', '上课出勤率', '作业完成率']].describe().reset_index()
    
    # 数据处理：按出勤率排序，计算平均作业完成率
    bigdata_sorted = bigdata_data.sort_values(by='上课出勤率')
    # 将出勤率分组，每5%为一组
    bigdata_sorted['出勤率分组'] = pd.cut(bigdata_sorted['上课出勤率'], bins=10, labels=False)
    bigdata_grouped = bigdata_sorted.groupby('出勤率分组').agg({
        '上课出勤率': 'mean',
        '作业完成率': 'mean'
    }).reset_index()
    
    # 创建两个图表
    # 第一个图表：大数据管理专业学生学习时长分布
    fig4_1 = px.histogram(bigdata_data, x='每周学习时长（小时）', title='大数据管理专业学习时长分布', nbins=20)
    
    # 第二个图表：大数据管理专业学生出勤率与作业完成率关系（折线图）
    fig4_2 = px.line(bigdata_grouped, x='上课出勤率', y='作业完成率', title='大数据管理专业出勤率与作业完成率关系')
    
    # 布局
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig4_1, use_container_width=True)
    with col2:
        st.plotly_chart(fig4_2, use_container_width=True)

elif nav == "测试成绩":
    st.title("🎓 期末成绩预测")
    st.write("输入学生的基本信息和平时表现，我们将为您预测期末成绩。")
    
    # 加载模型和编码器
    model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    le_major = joblib.load(os.path.join(model_dir, 'le_major.pkl'))
    le_gender = joblib.load(os.path.join(model_dir, 'le_gender.pkl'))
    
    # 创建预测表单
    st.subheader("🎯 成绩预测")
    with st.form("prediction_form"):
        # 学生信息输入
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input("学号")
            name = st.text_input("姓名")
            gender = st.selectbox("性别", ["男", "女"])
            major = st.selectbox("专业", le_major.classes_)
        
        with col2:
            study_hours = st.slider("每周学习时长（小时）", 0, 50, 20)
            attendance = st.slider("出勤率(%)", 0, 100, 95)
            midterm_grade = st.slider("期中考试成绩", 0, 100, 75)
            homework_completion = st.slider("作业完成率(%)", 0, 100, 90)
        
        # 提交按钮
        submit_button = st.form_submit_button("预测期末成绩", type="primary")
    
    # 预测结果
    if submit_button:
        # 数据预处理
        gender_encoded = le_gender.transform([gender])[0]
        major_encoded = le_major.transform([major])[0]
        
        # 准备预测数据
        input_data = pd.DataFrame({
            '性别': [gender_encoded],
            '专业': [major_encoded],
            '每周学习时长（小时）': [study_hours],
            '上课出勤率': [attendance / 100],
            '期中考试分数': [midterm_grade],
            '作业完成率': [homework_completion / 100]
        })
        
        # 预测
        predicted_grade = model.predict(input_data)[0]
        predicted_grade = int(round(predicted_grade))
        
        st.divider()
        st.header("📊 预测结果")
        
        # 显示预测成绩
        st.header(f"预测期末成绩: {predicted_grade}分")
        
        # 显示祝贺信息
        if predicted_grade >= 60:
            st.success("恭喜！您的预测成绩及格了！")
            st.image("guole.jpg", caption="庆祝一下！", width=500)
        else:
            st.warning("需要继续努力哦！")
            st.image("guake.jpg", caption="继续加油！", width=500)
    
    
