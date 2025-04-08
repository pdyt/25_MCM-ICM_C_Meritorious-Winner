import pandas as pd
import numpy as np
from datetime import datetime

def read_data():
    """读取原始数据文件"""
    df = pd.read_csv('oly/summerOly_athletes.csv', encoding='gbk')
    
    # 打印数据基本信息
    print("\n东道国列的基本信息:")
    print(f"数据类型: {df['东道国'].dtype}")
    print(f"唯一值: {df['东道国'].unique()}")
    print("值统计:\n", df['东道国'].value_counts(dropna=False))
    
    print("\n示例数据:")
    print(df[['NOC', '东道国', 'Year']].head(10))
    
    # 数据预处理
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['code'] = df['code'].astype(str)
    df['东道国'] = pd.to_numeric(df['东道国'], errors='coerce').fillna(0).astype(int)
    
    print(f"数据读取完成，共 {len(df)} 条记录")
    return df

def preprocess_data(df):
    """数据预处理"""
    # 确保时间列是整数格式
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    # 确保code列是字符串格式
    df['code'] = df['code'].astype(str)
    return df

def calculate_event_counts(group):
    """计算项目细分数量（该大类项目code下的细分项目event数量）"""
    return group['Event'].nunique()

def calculate_participation_stats(df, year, noc, sport_code):
    """计算参与统计数据"""
    # 该年所有奥运会参与人次
    year_total = len(df[df['Year'] == year])
    
    # 该年该项目代码的总参与人次
    year_sport_total = len(df[(df['Year'] == year) & (df['code'] == sport_code)])
    
    # 该年该国该项目的参与人次
    year_country_sport_total = len(df[(df['Year'] == year) & 
                                    (df['NOC'] == noc) & 
                                    (df['code'] == sport_code)])
    
    return pd.Series({
        'total_year_participations': year_total,
        'total_year_sport_participations': year_sport_total,
        'country_sport_participations': year_country_sport_total
    })

def calculate_historical_rates(group):
    """计算历史获奖频率（不包括当年）
    限定在特定国家和项目代码范围内
    每个运动员只计算一次最好的成绩
    """
    # 获取所有年份并排序
    years = sorted(group['Year'].unique())
    if len(years) < 2:  # 如果数据少于两年，返回0
        return pd.Series({
            '该国该项目运动员金牌获奖频率': 0,
            '该国该项目运动员银牌获奖频率': 0,
            '该国该项目运动员铜牌获奖频率': 0
        })
    
    # 获取当年和历史数据
    current_year = years[-1]
    historical_data = group[group['Year'] < current_year]
    
    # 计算唯一运动员数量
    total_athletes = len(historical_data['Name'].unique())
    if total_athletes == 0:
        return pd.Series({
            '该国该项目运动员金牌获奖频率': 0,
            '该国该项目运动员银牌获奖频率': 0,
            '该国该项目运动员铜牌获奖频率': 0
        })
    
    # 为每个运动员只保留最好的成绩
    # 创建奖牌等级映射
    medal_rank = {'Gold': 1, 'Silver': 2, 'Bronze': 3, 'No medal': 4, np.nan: 4}
    
    # 为每个运动员找到最好的成绩
    best_medals = historical_data.groupby('Name').agg({
        'Medal': lambda x: min((medal_rank.get(m, 4) for m in x), default=4)
    })
    
    # 统计每种奖牌的获得者数量（每个运动员只计算一次最好的成绩）
    medal_counts = pd.Series({
        'Gold': sum(best_medals['Medal'] == 1),
        'Silver': sum(best_medals['Medal'] == 2),
        'Bronze': sum(best_medals['Medal'] == 3)
    })
    
    # 计算频率
    return pd.Series({
        '该国该项目运动员金牌获奖频率': medal_counts['Gold'] / total_athletes,
        '该国该项目运动员银牌获奖频率': medal_counts['Silver'] / total_athletes,
        '该国该项目运动员铜牌获奖频率': medal_counts['Bronze'] / total_athletes
    })

def calculate_star_athletes(group):
    """计算明星运动员和新星运动员数量
    明星运动员：该项目该国家，上两届都有拿牌的运动员数量（同一个运动员只计算一次）
    新星运动员：在上一届获得2个及以上奖牌的运动员数量（同一个运动员只计算一次）
    """
    # 获取所有年份并排序
    years = sorted(group['Year'].unique())
    if len(years) < 3:  # 如果数据少于三年，返回0（需要当年和前两年的数据）
        return pd.Series({'明星运动员个数': 0, '新星运动员个数': 0})
    
    # 对每个年份计算
    result_list = []
    for current_year in years[2:]:  # 从第三年开始，确保有前两年的数据
        # 获取上两届的年份
        last_year = years[years.index(current_year) - 1]
        prev_year = years[years.index(current_year) - 2]
        
        # 获取上两届的数据
        last_year_data = group[group['Year'] == last_year]
        prev_year_data = group[group['Year'] == prev_year]
        
        # 获取每届获得过奖牌的运动员名单（去重）
        last_year_medalists = set(last_year_data[last_year_data['Medal'].notna() & (last_year_data['Medal'] != 'No medal')]['Name'].unique())
        prev_year_medalists = set(prev_year_data[prev_year_data['Medal'].notna() & (prev_year_data['Medal'] != 'No medal')]['Name'].unique())
        
        # 明星运动员：上两届都获得过奖牌的运动员（取交集）
        star_athletes = len(last_year_medalists.intersection(prev_year_medalists))
        
        # 计算新星运动员：上一届获得2个及以上奖牌的运动员
        last_year_medal_counts = last_year_data[last_year_data['Medal'].notna() & (last_year_data['Medal'] != 'No medal')].groupby('Name').size()
        rising_stars = len(last_year_medal_counts[last_year_medal_counts >= 2])
        
        result_list.append({
            'Year': current_year,
            'star_athletes': star_athletes,
            'rising_stars': rising_stars
        })
    
    # 如果没有计算结果，返回0
    if not result_list:
        return pd.Series({'明星运动员个数': 0, '新星运动员个数': 0})
    
    # 返回最后一年的结果
    last_result = result_list[-1]
    return pd.Series({
        '明星运动员个数': last_result['star_athletes'],
        '新星运动员个数': last_result['rising_stars']
    })

def calculate_career_length(group):
    """计算平均运动生涯时间"""
    if len(group) == 0:
        return 0
    
    # 获取每个运动员的参与年份
    athlete_years = group.groupby('Name')['Year'].agg(['min', 'max'])
    
    # 计算生涯时间
    career_lengths = athlete_years['max'] - athlete_years['min']
    
    # 过滤掉异常数据（超过24年）
    valid_lengths = career_lengths[career_lengths <= 24]
    
    # 如果是1896年（第一届现代奥运会），所有运动员的生涯时间都是0
    if athlete_years['min'].min() == 1896:
        return 0
    
    return valid_lengths.mean() if len(valid_lengths) > 0 else 0

def calculate_medal_counts(group, year, noc, sport_code):
    """计算当年实际获奖数量
    限定在特定国家和项目代码范围内
    """
    # 只考虑当年该国该项目的数据
    current_data = group[
        (group['Year'] == year) & 
        (group['NOC'] == noc) & 
        (group['code'] == sport_code)
    ]
    
    medals = current_data['Medal'].value_counts()
    return pd.Series({
        'gold_count': medals.get('Gold', 0),
        'silver_count': medals.get('Silver', 0),
        'bronze_count': medals.get('Bronze', 0)
    })

def process_data():
    """处理数据并生成统计结果"""
    print("开始读取数据文件...")
    # 读取数据文件
    df = pd.read_csv('./summerOly_athletes.csv', encoding='gbk')
    print(f"数据读取完成，共 {len(df)} 条记录")
    
    # 数据预处理，转换数据类型
    print("数据预处理...")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['code'] = df['code'].astype(str)
    df['东道国'] = pd.to_numeric(df['东道国'], errors='coerce').fillna(0).astype(int)
    
    print(f"数据读取完成，共 {len(df)} 条记录")
    
    # 预先计算东道主数据
    print("计算东道主数据...")
    host_data = df[df['东道国'] == 1][['Year', 'NOC']].drop_duplicates()
    host_dict = {(row.Year, row.NOC): 1 for row in host_data.itertuples()}
    
    # 一次性计算所有基本统计量
    print("计算基本统计量...")
    stats = df.groupby(['Year', 'NOC', 'code']).agg({
        'Name': 'count',  # 参与人次
        'Medal': [
            ('gold', lambda x: (x == 'Gold').sum()),
            ('silver', lambda x: (x == 'Silver').sum()),
            ('bronze', lambda x: (x == 'Bronze').sum())
        ],
        'Event': 'nunique'  # 项目细分数量
    }).reset_index()
    
    # 重命名列
    stats.columns = ['时间', '国家代码', '大类项目代码', '某年某国某大类参与人次', 
                    '实际当年其国某项目金牌数量', '实际当年其国某项目银牌数量', '实际当年其国某项目铜牌数量',
                    '该大类项目的细分项目event数量']
    
    # 计算总体参与人次（使用transform避免merge）
    print("计算参与人次...")
    year_participation = df.groupby('Year')['Name'].count()
    stats['某年整个奥林匹克参与人次'] = stats['时间'].map(year_participation)
    
    # 计算项目参与人次（使用transform避免merge）
    stats['某年整个大类参与人次'] = stats.groupby(['时间', '大类项目代码'])['某年某国某大类参与人次'].transform('sum')
    
    # 添加是否为东道国（使用字典查找代替apply）
    print("添加东道国标记...")
    stats['是否为东道国'] = stats.apply(lambda row: 1 if (row['时间'], row['国家代码']) in host_dict else 0, axis=1)
    
    # 获取所有奥运会年份并排序
    all_years = sorted(df['Year'].unique())
    
    # 计算明星运动员和新星运动员
    print("计算明星和新星运动员...")
    star_athletes = []
    rising_stars = []
    
    # 预先计算每个运动员在每届奥运会的奖牌情况
    print("预处理奖牌数据...")
    # 创建一个字典来存储每个运动员在每届奥运会的奖牌数量
    athlete_medals_dict = {}
    for _, row in df.iterrows():
        if pd.notna(row['Medal']) and row['Medal'] != 'No medal':
            key = (row['NOC'], row['code'], row['Name'], row['Year'])
            athlete_medals_dict[key] = athlete_medals_dict.get(key, 0) + 1
    
    # 对每一行计算明星运动员和新星运动员
    for _, row in stats.iterrows():
        year = row['时间']
        noc = row['国家代码']
        code = row['大类项目代码']
        
        try:
            year_idx = all_years.index(year)
            # 如果是前两届，无法计算前两届的数据，因此设置为0
            if year_idx < 2:  
                star_athletes.append(0)
                rising_stars.append(0)
                continue
            
            # 获取前两届年份
            last_year = all_years[year_idx - 1]
            prev_year = all_years[year_idx - 2]
            
            # 获取所有运动员在前两届的奖牌情况
            athletes_medals = {}  # 存储每个运动员的奖牌信息
            for athlete_key, medal_count in athlete_medals_dict.items():
                athlete_noc, athlete_code, athlete_name, athlete_year = athlete_key
                if athlete_noc == noc and athlete_code == code:
                    if athlete_year in [last_year, prev_year]:
                        if athlete_name not in athletes_medals:
                            athletes_medals[athlete_name] = {last_year: 0, prev_year: 0}
                        athletes_medals[athlete_name][athlete_year] = medal_count
            
            # 明星运动员：在前两届都获得过奖牌的运动员
            star_count = sum(1 for athlete_data in athletes_medals.values() 
                           if athlete_data[last_year] > 0 and athlete_data[prev_year] > 0)
            star_athletes.append(star_count)
            
            # 新星运动员：在上一届获得2个及以上奖牌的运动员
            rising_count = sum(1 for athlete_data in athletes_medals.values() 
                             if athlete_data[last_year] >= 2)
            rising_stars.append(rising_count)
            
        except Exception as e:
            print(f"Error processing {year}, {noc}, {code}: {str(e)}")
            star_athletes.append(0)
            rising_stars.append(0)
    
    stats['明星运动员个数'] = star_athletes
    stats['新星运动员个数'] = rising_stars
    
    # 计算历史获奖频率
    print("计算历史获奖频率...")
    rates = df.groupby(['NOC', 'code']).apply(calculate_historical_rates).reset_index()
    rates.columns = ['国家代码', '大类项目代码', '该国该项目运动员金牌获奖频率', 
                    '该国该项目运动员银牌获奖频率', '该国该项目运动员铜牌获奖频率']
    
    # 计算平均运动生涯时间
    print("计算运动生涯时间...")
    career = df.groupby(['NOC', 'code']).apply(calculate_career_length).reset_index()
    career.columns = ['国家代码', '大类项目代码', '平均运动生涯时间']
    
    # 合并所有结果（减少merge次数）
    print("合并结果...")
    result = stats.merge(rates, on=['国家代码', '大类项目代码'], how='left')
    result = result.merge(career, on=['国家代码', '大类项目代码'], how='left')
    
    # 确保所有列都存在并按顺序排列
    columns_order = [
        '时间', '国家代码', '大类项目代码', '是否为东道国',
        '某年整个奥林匹克参与人次', '某年整个大类参与人次', '某年某国某大类参与人次',
        '该大类项目的细分项目event数量', '该国该项目运动员金牌获奖频率',
        '该国该项目运动员银牌获奖频率', '该国该项目运动员铜牌获奖频率',
        '明星运动员个数', '新星运动员个数', '平均运动生涯时间',
        '实际当年其国某项目金牌数量', '实际当年其国某项目银牌数量', '实际当年其国某项目铜牌数量'
    ]
    
    # 填充缺失值并排序列
    result = result.fillna(0)[columns_order]
    
    # 保存结果
    print("保存结果...")
    result.to_csv('oly/stats_output.csv', index=False, encoding='gbk')
    
    print(f"\n处理完成！结果已保存到 oly/stats_output.csv")
    print(f"共生成 {len(result)} 条统计记录")

if __name__ == '__main__':
    process_data() 