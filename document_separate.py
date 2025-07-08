import os
from pathlib import Path
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter

def separate_pdf_pages(pdf_path, output_dir=None):
    """
    PDF 파일을 한 페이지씩 분할하여 저장하는 함수
    
    Args:
        pdf_path (str): 분할할 PDF 파일 경로
        output_dir (str): 출력 디렉토리 경로 (None이면 원본 파일과 같은 디렉토리)
    
    Returns:
        list: 생성된 파일 경로 리스트
    """
    try:
        # 입력 파일 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir) / "separated_pages"
        output_path.mkdir(exist_ok=True)
        
        # PDF 파일 읽기
        print(f"📄 PDF 파일을 읽는 중: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"📊 총 페이지 수: {total_pages}")
            
            created_files = []
            
            # 각 페이지를 개별 PDF로 저장
            for page_num in range(total_pages):
                # 새 PDF 작성기 생성
                pdf_writer = PdfWriter()
                
                # 현재 페이지 추가
                pdf_writer.add_page(pdf_reader.pages[page_num])
                
                # 출력 파일명 생성 (원본 파일명_페이지번호.pdf)
                original_name = Path(pdf_path).stem
                output_filename = f"{original_name}_page_{page_num + 1:03d}.pdf"
                output_file_path = output_path / output_filename
                
                # 파일 저장
                with open(output_file_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                created_files.append(str(output_file_path))
                print(f"✅ 페이지 {page_num + 1}/{total_pages} 저장 완료: {output_filename}")
            
            print(f"\n🎉 PDF 분할 완료!")
            print(f"📁 출력 디렉토리: {output_path}")
            print(f"📄 생성된 파일 수: {len(created_files)}")
            
            return created_files
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return []

def main():
    """메인 함수"""
    # 현재 스크립트가 있는 디렉토리
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # PDF 파일 경로
    pdf_file_path = os.path.join(current_dir, "CAIKE-TR-D271-사용자매뉴얼-v1.0_20241018.pdf")
    
    print("🚀 CAIKE 매뉴얼 PDF 분할 시작")
    print("=" * 50)
    
    # PDF 파일 존재 확인
    if not os.path.exists(pdf_file_path):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_file_path}")
        print("📁 현재 디렉토리의 파일 목록:")
        for file in os.listdir(current_dir):
            if file.endswith('.pdf'):
                print(f"   - {file}")
        return
    
    # PDF 분할 실행
    created_files = separate_pdf_pages(pdf_file_path)
    
    if created_files:
        print("\n📋 생성된 파일 목록:")
        for i, file_path in enumerate(created_files[:5], 1):  # 처음 5개만 표시
            print(f"   {i}. {os.path.basename(file_path)}")
        
        if len(created_files) > 5:
            print(f"   ... 및 {len(created_files) - 5}개 파일 더")
        
        print(f"\n💡 모든 파일은 'separated_pages' 폴더에 저장되었습니다.")
    else:
        print("\n❌ 파일 분할에 실패했습니다.")

if __name__ == "__main__":
    main()