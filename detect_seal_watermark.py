#!/usr/bin/env python3

import videoseal
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import pandas as pd
import os

def detect_watermark(model, img_path, N=5):
    """
    Detecta watermark em uma imagem
    Returns: (detected, confidence, binary_message)
    """
    try:
        img = Image.open(img_path)

        # Converter para RGB se necessário
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Converter para tensor
        img_tensor = T.ToTensor()(img).unsqueeze(0)

        # Detectar watermark
        # Testar N vezes para obter média e desvio padrão da confiança
        tests = []
        for _ in range(N):
            # Detectar
            detected = model.detect(img_tensor)

            # CORREÇÃO: preds[0,0] não é confiável!
            # Usar a magnitude média dos bits da mensagem (preds[0, 1:])
            # Imagens com watermark têm bits com valores absolutos MUITO maiores
            message_bits = detected["preds"][0, 1:]  # 256 bits da mensagem
            avg_magnitude = message_bits.abs().mean().item()  # Magnitude média

            tests.append(avg_magnitude)

        # Média dos resultados
        avg_watermark_magnitude = sum(tests) / len(tests)

        # Threshold baseado na análise:
        # - Imagens com watermark: magnitude média ~5-10
        # - Imagens sem watermark: magnitude média ~0.3-0.8
        # Threshold conservador em 2.0
        is_detected = avg_watermark_magnitude > 2.0
        confidence = avg_watermark_magnitude

        return is_detected, confidence

    except Exception as e:
        print(f"Erro ao processar {img_path}: {e}")
        return None, None, None

def evaluate_directory(model, img_dir, label="unknown"):
    """
    Avalia todas as imagens em um diretório
    """
    img_path = Path(img_dir)
    images = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))

    results = []

    for img_file in images:
        detected, confidence = detect_watermark(model, img_file)

        if detected is not None:
            results.append({
                'image': img_file.name,
                'directory': label,
                'detected': detected,
                'confidence': confidence
            })

    return results

def run_evaluation(model):
    """
    Executa avaliação completa conforme pipeline
    """
    all_results = []

    print("="*60)
    print("Avaliação de Robustez do Watermark")
    print("="*60 + "\n")

    # 1. Baseline negativo (imagens reais sem watermark)
    print("1. Baseline Negativo - Imagens Reais (Flickr8)")
    flickr_dir = Path('flickr8/Images')
    if flickr_dir.exists():
        # Testar apenas uma amostra (mesmas 20 imagens)
        import random
        random.seed(42)
        all_flickr = list(flickr_dir.glob('*.jpg'))
        sample_flickr = random.sample(all_flickr, min(20, len(all_flickr)))

        for img_file in sample_flickr:
            detected, confidence = detect_watermark(model, img_file)
            if detected is not None:
                all_results.append({
                    'image': img_file.name,
                    'directory': 'flickr8_original',
                    'attack': 'none',
                    'detected': detected,
                    'confidence': confidence,
                    'expected': False  # Não deve ser detectado
                })
        print(f"   ✓ Processadas {len(sample_flickr)} imagens\n")
    else:
        print(f"   ✗ Diretório {flickr_dir} não encontrado\n")

    # 2. Verdadeiros positivos - Imagens com watermark (sem ataque)
    print("2. Verdadeiros Positivos - Imagens com Watermark")
    seal_dir = Path('seal')
    if seal_dir.exists():
        seal_results = evaluate_directory(model, seal_dir, 'seal_original')
        for res in seal_results:
            res['attack'] = 'none'
            res['expected'] = True  # Deve ser detectado
            all_results.append(res)
        print(f"   ✓ Processadas {len(seal_results)} imagens\n")
    else:
        print(f"   ✗ Diretório {seal_dir} não encontrado\n")

    # 3. Teste de robustez - Imagens com ataques
    print("3. Teste de Robustez - Imagens com Ataques")
    attacks_dir = Path('seal_attacks')
    if attacks_dir.exists():
        attack_types = [d.name for d in attacks_dir.iterdir() if d.is_dir()]

        for attack in sorted(attack_types):
            attack_path = attacks_dir / attack
            attack_results = evaluate_directory(model, attack_path, f'seal_{attack}')

            for res in attack_results:
                res['attack'] = attack
                res['expected'] = True  # Ainda deve ser detectado (teste de robustez)
                all_results.append(res)

            detected_count = sum(1 for r in attack_results if r['detected'])
            total = len(attack_results)
            rate = (detected_count / total * 100) if total > 0 else 0
            print(f"   {attack:20s}: {detected_count}/{total} detectadas ({rate:.1f}%)")

        print()
    else:
        print(f"   ✗ Diretório {attacks_dir} não encontrado\n")

    # 4. Teste de robustez - Ataques compostos
    print("4. Teste de Robustez - Ataques Compostos (múltiplas transformações)")
    combined_attacks_dir = Path('seal_combined_attacks')
    if combined_attacks_dir.exists():
        scenario_types = [d.name for d in combined_attacks_dir.iterdir() if d.is_dir()]

        for scenario in sorted(scenario_types):
            scenario_path = combined_attacks_dir / scenario
            scenario_results = evaluate_directory(model, scenario_path, f'seal_{scenario}')

            for res in scenario_results:
                res['attack'] = f'combined_{scenario}'
                res['expected'] = True
                all_results.append(res)

            detected_count = sum(1 for r in scenario_results if r['detected'])
            total = len(scenario_results)
            rate = (detected_count / total * 100) if total > 0 else 0
            print(f"   {scenario:25s}: {detected_count}/{total} detectadas ({rate:.1f}%)")

        print()
    else:
        print(f"   ⚠ Diretório {combined_attacks_dir} não encontrado (execute apply_combined_attacks.py)\n")

    # Criar DataFrame com resultados
    df = pd.DataFrame(all_results)

    # Salvar resultados completos
    output_file = 'watermark_detection_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Resultados salvos em: {output_file}\n")

    # Gerar relatório resumido
    generate_report(df)

    return df

def generate_report(df):
    """
    Gera relatório resumido com métricas
    """
    print("="*60)
    print("RELATÓRIO DE RESULTADOS")
    print("="*60 + "\n")

    # Métricas gerais
    if 'expected' in df.columns:
        # 1. Falsos Positivos (imagens reais detectadas como watermark)
        negatives = df[df['expected'] == False]
        if len(negatives) > 0:
            fp_count = negatives['detected'].sum()
            fp_rate = (fp_count / len(negatives)) * 100
            print(f"1. Taxa de Falso Positivo (imagens reais):")
            print(f"   {fp_count}/{len(negatives)} detectadas incorretamente ({fp_rate:.1f}%)")
            print(f"   {'✓ Ótimo' if fp_rate < 5 else '⚠ Atenção'}\n")

        # 2. Verdadeiros Positivos (sem ataque)
        original = df[(df['attack'] == 'none') & (df['expected'] == True)]
        if len(original) > 0:
            tp_count = original['detected'].sum()
            tp_rate = (tp_count / len(original)) * 100
            print(f"2. Taxa de Detecção (imagens com watermark, sem ataque):")
            print(f"   {tp_count}/{len(original)} detectadas ({tp_rate:.1f}%)")
            print(f"   {'✓ Ótimo' if tp_rate > 95 else '⚠ Atenção'}\n")

    # 3. Robustez por tipo de ataque
    attacked = df[df['attack'] != 'none']
    if len(attacked) > 0:
        # Separar ataques simples e compostos
        simple_attacks = attacked[~attacked['attack'].str.startswith('combined_')]
        combined_attacks = attacked[attacked['attack'].str.startswith('combined_')]

        if len(simple_attacks) > 0:
            print(f"3. Robustez - Ataques Simples:")
            print("-" * 60)

            attack_summary = simple_attacks.groupby('attack').agg({
                'detected': ['sum', 'count', 'mean']
            }).round(3)

            attack_summary.columns = ['Detectadas', 'Total', 'Taxa']
            attack_summary['Taxa %'] = (attack_summary['Taxa'] * 100).round(1)
            attack_summary = attack_summary.sort_values('Taxa', ascending=False)

            print(attack_summary[['Detectadas', 'Total', 'Taxa %']].to_string())
            print()

        if len(combined_attacks) > 0:
            print(f"\n4. Robustez - Ataques Compostos (múltiplas transformações):")
            print("-" * 60)

            # Remover prefixo 'combined_' para melhor visualização
            combined_attacks_clean = combined_attacks.copy()
            combined_attacks_clean['attack'] = combined_attacks_clean['attack'].str.replace('combined_', '')

            combined_summary = combined_attacks_clean.groupby('attack').agg({
                'detected': ['sum', 'count', 'mean']
            }).round(3)

            combined_summary.columns = ['Detectadas', 'Total', 'Taxa']
            combined_summary['Taxa %'] = (combined_summary['Taxa'] * 100).round(1)
            combined_summary = combined_summary.sort_values('Taxa', ascending=False)

            print(combined_summary[['Detectadas', 'Total', 'Taxa %']].to_string())
            print()

        # Classificar todos os ataques por severidade
        print(f"\n5. Classificação Geral de Ataques por Impacto:")
        print("-" * 60)

        all_attack_summary = attacked.groupby('attack').agg({
            'detected': ['sum', 'count', 'mean']
        }).round(3)
        all_attack_summary.columns = ['Detectadas', 'Total', 'Taxa']
        all_attack_summary['Taxa %'] = (all_attack_summary['Taxa'] * 100).round(1)
        all_attack_summary = all_attack_summary.sort_values('Taxa', ascending=False)

        for idx, row in all_attack_summary.iterrows():
            rate = row['Taxa %']
            impact = "BAIXO" if rate > 80 else "MÉDIO" if rate > 50 else "ALTO"
            emoji = "✓" if rate > 80 else "⚠" if rate > 50 else "✗"
            attack_name = idx.replace('combined_', '🔗 ')
            print(f"   {emoji} {attack_name:30s}: {rate:5.1f}% - Impacto {impact}")

        # Comparação: ataques simples vs compostos
        if len(simple_attacks) > 0 and len(combined_attacks) > 0:
            print(f"\n6. Comparação: Ataques Simples vs Compostos:")
            print("-" * 60)
            simple_avg = simple_attacks['detected'].mean() * 100
            combined_avg = combined_attacks['detected'].mean() * 100
            print(f"   Taxa média - Ataques Simples:   {simple_avg:.1f}%")
            print(f"   Taxa média - Ataques Compostos: {combined_avg:.1f}%")
            print(f"   Diferença: {simple_avg - combined_avg:+.1f}% pontos percentuais")
            if combined_avg < simple_avg - 10:
                print(f"   ⚠ Ataques compostos são significativamente mais efetivos!")
            elif combined_avg < simple_avg:
                print(f"   → Ataques compostos reduzem a taxa de detecção")
            else:
                print(f"   ✓ Watermark robusto até para ataques compostos")

    print("\n" + "="*60)

if __name__ == '__main__':
    print("Carregando modelo Seal...")

    # Carregar modelo
    model_card_path = Path(os.path.dirname(videoseal.__file__)) / 'cards' / 'videoseal_1.0.yaml'
    model = videoseal.load(model_card_path)

    print("Modelo carregado com sucesso!\n")

    # Executar avaliação
    results_df = run_evaluation(model)
