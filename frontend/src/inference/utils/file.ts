export function openFileDialog(accept: string): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = accept
    input.onchange = () => {
      resolve(input.files?.[0] ?? null)
    }
    input.oncancel = () => resolve(null)
    input.click()
  })
}
