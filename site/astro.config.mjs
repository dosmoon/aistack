import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://dosmoon.com',
  base: '/aistack',
  integrations: [
    starlight({
      title: 'aistack',
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        'zh-cn': { label: '简体中文', lang: 'zh-CN' },
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/dosmoon/aistack' },
      ],
      lastUpdated: true,
    }),
  ],
});
