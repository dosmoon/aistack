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
      sidebar: [
        { label: 'aistack', link: '/' },
        { label: 'Integration Guide', link: '/integration/' },
        { label: 'Configuration', link: '/configuration/' },
        { label: 'HTTP API', items: [{ autogenerate: { directory: 'api' } }] },
        { label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
        { label: 'Research Notes', items: [{ autogenerate: { directory: 'research' } }] },
      ],
      lastUpdated: true,
    }),
  ],
});
