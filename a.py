import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# 全局样式：深色背景+白色文字（匹配参考图）
st.markdown("""
    <style>
    .stApp {background-color: #121212; color: #fff;}
    .stDataFrame {background-color: #1e1e1e; color: #fff; border: none;}
    .stMetric {background-color: #1e1e1e; color: #fff;}
    .stSubheader {color: #fff;}
    .stTitle {color: #fff;}
    .stTextInput, .stSelectbox, .stSlider {background-color: #1e1e1e; color: #fff; border: 1px solid #333;}
    .stButton > button {background-color: #ff4b4b; color: #fff; border: none;}
    </style>
""", unsafe_allow_html=True)

# 设置页面配置
st.set_page_config(
    page_title="专业数据分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏导航
st.sidebar.title("导航菜单")
menu = st.sidebar.radio(
    "选择功能",
    ["项目介绍", "专业数据分析", "成绩预测"]
)


# 封装数据读取函数
def get_dataframe_from_csv():
    csv_file_path = "student_data_adjusted_rounded.csv"
    try:
        df = pd.read_csv(csv_file_path)
        df["上课出勤率(%)"] = (df["上课出勤率"] * 100).round(1)
        return df
    except FileNotFoundError:
        st.error(f"❌ 未找到文件：{csv_file_path}")
        st.stop()
    except Exception as e:
        st.error(f"❌ 读取数据出错：{str(e)}")
        st.stop()


# 项目介绍页面（保持不变）
if menu == "项目介绍":
    st.title("学生成绩分析与预测系统")
    st.markdown("---")
    st.subheader("📋 项目概述")
    st.write("本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。")
    st.subheader("✨ 主要特点")
    st.markdown("""
    - **数据可视化**：多维度展示学生学业数据
    - **专业分析**：按专业学习的详细成绩分析
    - **智能预测**：基于机器学习模型的成绩预测
    - **学习建议**：根据预测结果提供个性化反馈
    """)
    st.subheader("🎯 项目目标")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 目标一：分析影响因素")
        st.write("- 识别关键学习指标\n- 探索成绩相关因素\n- 提供改进支持决策")
    with col2:
        st.markdown("#### 目标二：可视化展示")
        st.write("- 专业对比分析\n- 性别差异研究\n- 学习状况识别")
    with col3:
        st.markdown("#### 目标三：成绩预测")
        st.write("- 机器学习模型\n- 个性化预测\n- 及时干预预警")
    st.subheader("🔧 技术架构")
    arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)
    with arch_col1:
        st.markdown("**前端框架**")
        st.write("Streamlit")
    with arch_col2:
        st.markdown("**数据处理**")
        st.write("Pandas\nNumpy")
    with arch_col3:
        st.markdown("**可视化**")
        st.write("Plotly\nMatplotlib")
    with arch_col4:
        st.markdown("**机器学习**")
        st.write("Scikit-learn")


# 专业数据分析页面（完全复刻参考图）
elif menu == "专业数据分析":
    st.title("专业数据分析")
    st.markdown("---")
    df = get_dataframe_from_csv()  
    
    # 按专业分组统计（基础数据）
    major_stats = df.groupby("专业").agg({
        "每周学习时长（小时）": "mean",
        "期中考试分数": "mean",
        "期末考试分数": "mean",
        "上课出勤率(%)": "mean",
        "学号": "count"
    }).round(2).rename(columns={"学号": "人数"}).reset_index()


    # ========== 1. 各专业男女性别比例 ==========
    st.subheader("1. 各专业男女性别比例")
    col1, col2 = st.columns([2, 1])
    
    # 性别数据统计
    gender_data = df.groupby(["专业", "性别"]).size().reset_index(name="人数")
    with col1:
        # 双层柱状图（匹配参考图配色）
        fig_gender = px.bar(
            gender_data,
            x="专业",
            y="人数",
            color="性别",
            barmode="group",
            color_discrete_map={"男": "#1a73e8", "女": "#e3f2fd"},
            title="各专业男女性别分布",
            height=200
        )
        fig_gender.update_layout(
            showlegend=False,
            xaxis_title="", yaxis_title="人数",
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font_color="#fff",
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # 性别比例数据表格
        st.markdown("**性别比例数据**")
        gender_pivot = gender_data.pivot(index="专业", columns="性别", values="人数").fillna(0).astype(int)
        st.dataframe(gender_pivot, use_container_width=True)


    # ========== 2. 各专业学习指标对比 ==========
    st.subheader("2. 各专业学习指标对比")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 堆叠图+双折线（匹配参考图）
        fig_metrics = go.Figure()
        # 学习时长（背景柱形）
        fig_metrics.add_trace(go.Bar(
            x=major_stats["专业"],
            y=major_stats["每周学习时长（小时）"],
            name="学习时长",
            marker_color="#e3f2fd",
            opacity=0.8
        ))
        # 期中分数（绿色折线）
        fig_metrics.add_trace(go.Scatter(
            x=major_stats["专业"],
            y=major_stats["期中考试分数"],
            name="期中分数",
            mode="lines",
            line=dict(color="#2ecc71", width=3),
            yaxis="y2"
        ))
        # 期末分数（橙色折线）
        fig_metrics.add_trace(go.Scatter(
            x=major_stats["专业"],
            y=major_stats["期末考试分数"],
            name="期末分数",
            mode="lines",
            line=dict(color="#f39c12", width=3),
            yaxis="y2"
        ))
        fig_metrics.update_layout(
            title="各专业学习指标对比",
            xaxis_title="",
            yaxis_title="学习时长(小时)",
            yaxis2=dict(title="分数", overlaying="y", side="right"),
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font_color="#fff",
            showlegend=False,
            height=200,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # 详细数据表格
        st.markdown("**详细数据**")
        metrics_table = major_stats[["专业", "每周学习时长（小时）", "期中考试分数", "期末考试分数"]]
        st.dataframe(metrics_table, use_container_width=True)


    # ========== 3. 各专业出勤率分析 ==========
    st.subheader("3. 各专业出勤率分析")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 色块图（匹配参考图配色）
        fig_attendance = px.bar(
            major_stats,
            x="专业",
            y="人数",
            color="上课出勤率(%)",
            color_continuous_scale=px.colors.sequential.YlGnBu,
            title="各专业出勤率分布",
            height=200
        )
        fig_attendance.update_layout(
            xaxis_title="", yaxis_title="人数",
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font_color="#fff",
            coloraxis_showscale=True,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_attendance, use_container_width=True)
    
    with col2:
        # 出勤率排名表格
        st.markdown("**出勤率排名**")
        attendance_rank = major_stats.sort_values("上课出勤率(%)", ascending=False)[["专业", "上课出勤率(%)"]]
        attendance_rank["排名"] = range(1, len(attendance_rank)+1)
        st.dataframe(attendance_rank[["排名", "专业", "上课出勤率(%)"]], use_container_width=True)


    # ========== 4. 大数据管理专业专项分析 ==========
    st.subheader("4. 大数据管理专业专项分析")
    bigdata_df = df[df["专业"] == "大数据管理"]
    bigdata_avg = bigdata_df.agg({
        "上课出勤率(%)": "mean",
        "期末考试分数": "mean",
        "作业完成率": lambda x: (x.mean()*100).round(1),
        "每周学习时长（小时）": "mean"
    }).round(1)
    
    # 指标卡（匹配参考图）
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"平均出勤率：{bigdata_avg['上课出勤率(%)']}%")
    with col2:
        st.write(f"平均期末分数：{bigdata_avg['期末考试分数']}分")
    with col3:
        st.write(f"作业完成率：{bigdata_avg['作业完成率']}%")
    with col4:
        st.write(f"每周学习时长：{bigdata_avg['每周学习时长（小时）']}小时")
    
    # 大数据专业数据分布
    col1, col2 = st.columns(2)
    with col1:
        fig_bigdata_score = px.bar(
            bigdata_df,
            x="学号",
            y="期末考试分数",
            title="大数据管理专业学生期末分数分布",
            color="期末考试分数",
            color_continuous_scale=px.colors.sequential.Teal,
            height=200
        )
        fig_bigdata_score.update_layout(
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font_color="#fff",
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_bigdata_score, use_container_width=True)
    
    with col2:
        fig_bigdata_attendance = px.bar(
            bigdata_df,
            x="学号",
            y="上课出勤率(%)",
            title="大数据管理专业学生出勤率分布",
            color="上课出勤率(%)",
            color_continuous_scale=px.colors.sequential.Greens,
            height=200
        )
        fig_bigdata_attendance.update_layout(
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font_color="#fff",
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_bigdata_attendance, use_container_width=True)


# 成绩预测页面（完全复刻参考图）
elif menu == "成绩预测":
    st.title("期末成绩预测")
    st.markdown("---")
    st.write("请输入学生的学习信息，系统将预测其期末成绩并提供学习建议")

    # 输入表单（匹配参考图）
    with st.form("prediction_form"):
        # 学号输入
        student_id = st.text_input("学号", value="12321321")
        # 性别选择
        gender = st.selectbox("性别", options=["男", "女"], index=0)
        # 专业选择（读取CSV中的专业列表）
        df = get_dataframe_from_csv()
        major_list = df["专业"].unique().tolist()
        major = st.selectbox("专业", options=major_list, index=major_list.index("信息系统") if "信息系统" in major_list else 0)
        # 各项指标滑块
        study_time = st.slider("每周学习时长(小时)", min_value=0, max_value=40, value=10)
        attendance = st.slider("上课出勤率", min_value=0.0, max_value=1.0, value=0.9)
        midterm_score = st.slider("期中考试分数", min_value=0, max_value=100, value=60)
        homework_rate = st.slider("作业完成率", min_value=0.0, max_value=1.0, value=0.7)
        
        # 预测按钮
        submit_btn = st.form_submit_button("预测期末成绩", type="primary")

    # 预测结果展示
    if submit_btn:
        # 计算预测分数（可替换为真实模型）
        predict_score = 0.2*study_time + 0.2*attendance*100 + 0.4*midterm_score + 0.2*homework_rate*100
        predict_score = round(predict_score, 1)
        predict_score = max(min(predict_score, 100), 0)  # 限制在0-100之间

        # 展示预测结果
        st.subheader("预测结果")
        st.success(f"预测期末成绩：{predict_score}分")
        
        # 展示祝贺图（匹配参考图）
        if predict_score >= 60:
            st.image("https://img.lovepik.com/element/40087/7634.png_860.png", caption="恭喜！预测结果显示你合格啦！")
        else:
            st.warning("建议增加学习时长，提升出勤率和作业完成率哦~")
